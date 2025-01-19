use std::iter;

use base_db::CrateId;
use hir_def::db::DefDatabase;
use hir_def::generics::{WherePredicate, WherePredicateTypeTarget};
use hir_def::resolver::{HasResolver, TypeNs};
use hir_def::type_ref::{TraitBoundModifier, TypeRef};
use hir_def::{GenericDefId, TraitId, TypeAliasId, TypeOrConstParamId};
use hir_expand::name::Name;
use rustc_hash::FxHashSet;
use rustc_type_ir::inherent::IntoKind;
use rustc_type_ir::visit::TypeVisitableExt;
use smallvec::{smallvec, SmallVec};

use crate::db::HirDatabase;
use crate::lower_nextsolver::generic_predicates_for_param_query;

use super::{Binder, TraitRef};

// FIXME: use rustc_type_ir's elaborate

/// Returns an iterator over the whole super trait hierarchy (including the
/// trait itself).
pub fn all_super_traits(db: &dyn DefDatabase, trait_: TraitId) -> SmallVec<[TraitId; 4]> {
    // we need to take care a bit here to avoid infinite loops in case of cycles
    // (i.e. if we have `trait A: B; trait B: A;`)

    let mut result = smallvec![trait_];
    let mut i = 0;
    while let Some(&t) = result.get(i) {
        // yeah this is quadratic, but trait hierarchies should be flat
        // enough that this doesn't matter
        direct_super_traits(db, t, |tt| {
            if !result.contains(&tt) {
                result.push(tt);
            }
        });
        i += 1;
    }
    result
}

/// Given a trait ref (`Self: Trait`), builds all the implied trait refs for
/// super traits. The original trait ref will be included. So the difference to
/// `all_super_traits` is that we keep track of type parameters; for example if
/// we have `Self: Trait<u32, i32>` and `Trait<T, U>: OtherTrait<U>` we'll get
/// `Self: OtherTrait<i32>`.
pub fn all_super_trait_refs<T>(
    db: &dyn HirDatabase,
    trait_ref: TraitRef,
    cb: impl FnMut(TraitRef) -> Option<T>,
) -> Option<T> {
    let seen = iter::once(trait_ref.def_id).collect();
    SuperTraits { db, seen, stack: vec![trait_ref] }.find_map(cb)
}

struct SuperTraits<'a> {
    db: &'a dyn HirDatabase,
    stack: Vec<TraitRef>,
    seen: FxHashSet<GenericDefId>,
}

impl SuperTraits<'_> {
    fn elaborate(&mut self, trait_ref: &TraitRef) {
        direct_super_trait_refs(self.db, trait_ref, |trait_ref| {
            if !self.seen.contains(&trait_ref.def_id) {
                self.stack.push(trait_ref);
            }
        });
    }
}

impl Iterator for SuperTraits<'_> {
    type Item = TraitRef;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(next) = self.stack.pop() {
            self.elaborate(&next);
            Some(next)
        } else {
            None
        }
    }
}

fn direct_super_traits(db: &dyn DefDatabase, trait_: TraitId, cb: impl FnMut(TraitId)) {
    let resolver = trait_.resolver(db);
    let generic_params = db.generic_params(trait_.into());
    let trait_self = generic_params.trait_self_param();
    generic_params
        .where_predicates()
        .filter_map(|pred| match pred {
            WherePredicate::ForLifetime { target, bound, .. }
            | WherePredicate::TypeBound { target, bound } => {
                let is_trait = match target {
                    WherePredicateTypeTarget::TypeRef(type_ref) => {
                        match &generic_params.types_map[*type_ref] {
                            TypeRef::Path(p) => p.is_self_type(),
                            _ => false,
                        }
                    }
                    WherePredicateTypeTarget::TypeOrConstParam(local_id) => {
                        Some(*local_id) == trait_self
                    }
                };
                match is_trait {
                    true => bound.as_path(),
                    false => None,
                }
            }
            WherePredicate::Lifetime { .. } => None,
        })
        .filter(|(_, bound_modifier)| matches!(bound_modifier, TraitBoundModifier::None))
        .filter_map(|(path, _)| match resolver.resolve_path_in_type_ns_fully(db, path) {
            Some(TypeNs::TraitId(t)) => Some(t),
            _ => None,
        })
        .for_each(cb);
}

fn direct_super_trait_refs(db: &dyn HirDatabase, trait_ref: &TraitRef, cb: impl FnMut(TraitRef)) {
    let generic_params = db.generic_params(trait_ref.def_id);
    let trait_self = match generic_params.trait_self_param() {
        Some(p) => TypeOrConstParamId { parent: trait_ref.def_id, local_id: p },
        None => return,
    };
    generic_predicates_for_param_query(db, trait_self.parent, trait_self, None)
        .iter()
        .filter_map(|pred| {
            match pred.clone().kind().skip_binder() {
                rustc_type_ir::ClauseKind::Trait(trait_predicate) => {
                    let trait_ref = trait_predicate.trait_ref;
                    assert!(!trait_ref.has_escaping_bound_vars(), "FIXME unexpected higher-ranked trait bound");
                    Some(trait_ref)
                }
                _ => None,
            }
        })
        .for_each(cb);
}

pub fn associated_type_by_name_including_super_traits(
    db: &dyn HirDatabase,
    trait_ref: TraitRef,
    name: &Name,
) -> Option<(TraitRef, TypeAliasId)> {
    let fake_ir = crate::next_solver::DbIr::new(db, CrateId::from_raw(la_arena::RawIdx::from_u32(0)), None);
    rustc_type_ir::elaborate::supertraits(fake_ir, Binder::dummy(trait_ref)).find_map(|t| {
        let trait_id = match t.as_ref().skip_binder().def_id {
            GenericDefId::TraitId(id) => id,
            _ => unreachable!(),
        };
        let assoc_type = db.trait_data(trait_id).associated_type_by_name(name)?;
        Some((t.skip_binder(), assoc_type))        
    })
}
