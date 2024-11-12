//! This module contains the implementations of the `ToChalk` trait, which
//! handles conversion between our data types and their corresponding types in
//! Chalk (in both directions); plus some helper functions for more specialized
//! conversions.

use chalk_ir::interner::HasInterner;
use chalk_solve::rust_ir;

use base_db::ra_salsa::{self, InternKey};
use hir_def::{
    ConstParamId, LifetimeParamId, TraitId, TypeAliasId, TypeOrConstParamId, TypeParamId
};
use rustc_type_ir::inherent::{PlaceholderLike, SliceLike};

use crate::{
    chalk_db,
    db::HirDatabase,
    interner::rustc::{
        RustcAdtDef, RustcBoundRegion, RustcEarlyParamRegion, RustcErrorGuaranteed, RustcGenericArg, RustcGenericArgs, RustcInterner, RustcParamConst, RustcParamTy, RustcPlaceholderRegion, RustcRegion, RustcTy, RustcTys
    },
    AssocTypeId, CallableDefId, ChalkTraitId, FnDefId, ForeignDefId, Interner, OpaqueTyId,
    PlaceholderIndex,
};

pub(crate) trait ToChalk {
    type Chalk;
    fn to_chalk(self, db: &dyn HirDatabase) -> Self::Chalk;
    fn from_chalk(db: &dyn HirDatabase, chalk: Self::Chalk) -> Self;
}

pub(crate) fn from_chalk<T, ChalkT>(db: &dyn HirDatabase, chalk: ChalkT) -> T
where
    T: ToChalk<Chalk = ChalkT>,
{
    T::from_chalk(db, chalk)
}

impl ToChalk for hir_def::ImplId {
    type Chalk = chalk_db::ImplId;

    fn to_chalk(self, _db: &dyn HirDatabase) -> chalk_db::ImplId {
        chalk_ir::ImplId(self.as_intern_id())
    }

    fn from_chalk(_db: &dyn HirDatabase, impl_id: chalk_db::ImplId) -> hir_def::ImplId {
        InternKey::from_intern_id(impl_id.0)
    }
}

impl ToChalk for CallableDefId {
    type Chalk = FnDefId;

    fn to_chalk(self, db: &dyn HirDatabase) -> FnDefId {
        db.intern_callable_def(self).into()
    }

    fn from_chalk(db: &dyn HirDatabase, fn_def_id: FnDefId) -> CallableDefId {
        db.lookup_intern_callable_def(fn_def_id.into())
    }
}

pub(crate) struct TypeAliasAsValue(pub(crate) TypeAliasId);

impl ToChalk for TypeAliasAsValue {
    type Chalk = chalk_db::AssociatedTyValueId;

    fn to_chalk(self, _db: &dyn HirDatabase) -> chalk_db::AssociatedTyValueId {
        rust_ir::AssociatedTyValueId(self.0.as_intern_id())
    }

    fn from_chalk(
        _db: &dyn HirDatabase,
        assoc_ty_value_id: chalk_db::AssociatedTyValueId,
    ) -> TypeAliasAsValue {
        TypeAliasAsValue(TypeAliasId::from_intern_id(assoc_ty_value_id.0))
    }
}

impl From<FnDefId> for crate::db::InternedCallableDefId {
    fn from(fn_def_id: FnDefId) -> Self {
        InternKey::from_intern_id(fn_def_id.0)
    }
}

impl From<crate::db::InternedCallableDefId> for FnDefId {
    fn from(callable_def_id: crate::db::InternedCallableDefId) -> Self {
        chalk_ir::FnDefId(callable_def_id.as_intern_id())
    }
}

impl From<OpaqueTyId> for crate::db::InternedOpaqueTyId {
    fn from(id: OpaqueTyId) -> Self {
        InternKey::from_intern_id(id.0)
    }
}

impl From<crate::db::InternedOpaqueTyId> for OpaqueTyId {
    fn from(id: crate::db::InternedOpaqueTyId) -> Self {
        chalk_ir::OpaqueTyId(id.as_intern_id())
    }
}

impl From<chalk_ir::ClosureId<Interner>> for crate::db::InternedClosureId {
    fn from(id: chalk_ir::ClosureId<Interner>) -> Self {
        Self::from_intern_id(id.0)
    }
}

impl From<crate::db::InternedClosureId> for chalk_ir::ClosureId<Interner> {
    fn from(id: crate::db::InternedClosureId) -> Self {
        chalk_ir::ClosureId(id.as_intern_id())
    }
}

impl From<chalk_ir::CoroutineId<Interner>> for crate::db::InternedCoroutineId {
    fn from(id: chalk_ir::CoroutineId<Interner>) -> Self {
        Self::from_intern_id(id.0)
    }
}

impl From<crate::db::InternedCoroutineId> for chalk_ir::CoroutineId<Interner> {
    fn from(id: crate::db::InternedCoroutineId) -> Self {
        chalk_ir::CoroutineId(id.as_intern_id())
    }
}

pub fn to_foreign_def_id(id: TypeAliasId) -> ForeignDefId {
    chalk_ir::ForeignDefId(ra_salsa::InternKey::as_intern_id(&id))
}

pub fn from_foreign_def_id(id: ForeignDefId) -> TypeAliasId {
    ra_salsa::InternKey::from_intern_id(id.0)
}

pub fn to_assoc_type_id(id: TypeAliasId) -> AssocTypeId {
    chalk_ir::AssocTypeId(ra_salsa::InternKey::as_intern_id(&id))
}

pub fn from_assoc_type_id(id: AssocTypeId) -> TypeAliasId {
    ra_salsa::InternKey::from_intern_id(id.0)
}

pub fn from_placeholder_idx(db: &dyn HirDatabase, idx: PlaceholderIndex) -> TypeOrConstParamId {
    assert_eq!(idx.ui, chalk_ir::UniverseIndex::ROOT);
    let interned_id = ra_salsa::InternKey::from_intern_id(ra_salsa::InternId::from(idx.idx));
    db.lookup_intern_type_or_const_param_id(interned_id)
}

pub fn to_placeholder_idx(db: &dyn HirDatabase, id: TypeOrConstParamId) -> PlaceholderIndex {
    let interned_id = db.intern_type_or_const_param_id(id);
    PlaceholderIndex {
        ui: chalk_ir::UniverseIndex::ROOT,
        idx: ra_salsa::InternKey::as_intern_id(&interned_id).as_usize(),
    }
}

pub fn lt_from_placeholder_idx(db: &dyn HirDatabase, idx: PlaceholderIndex) -> LifetimeParamId {
    assert_eq!(idx.ui, chalk_ir::UniverseIndex::ROOT);
    let interned_id = ra_salsa::InternKey::from_intern_id(ra_salsa::InternId::from(idx.idx));
    db.lookup_intern_lifetime_param_id(interned_id)
}

pub fn lt_to_placeholder_idx(db: &dyn HirDatabase, id: LifetimeParamId) -> PlaceholderIndex {
    let interned_id = db.intern_lifetime_param_id(id);
    PlaceholderIndex {
        ui: chalk_ir::UniverseIndex::ROOT,
        idx: ra_salsa::InternKey::as_intern_id(&interned_id).as_usize(),
    }
}

pub fn ty_to_rustc_param_idx(db: &dyn HirDatabase, id: TypeParamId) -> RustcParamTy {
    let interned_id = db.intern_type_or_const_param_id(id.into());
    RustcParamTy { index: ra_salsa::InternKey::as_intern_id(&interned_id).as_u32() }
}

pub fn const_to_rustc_param_idx(db: &dyn HirDatabase, id: ConstParamId) -> RustcParamConst {
    let interned_id = db.intern_type_or_const_param_id(id.into());
    RustcParamConst { index: ra_salsa::InternKey::as_intern_id(&interned_id).as_u32() }
}

pub fn lt_to_rustc_param_idx(db: &dyn HirDatabase, id: LifetimeParamId) -> RustcEarlyParamRegion {
    let interned_id = db.intern_lifetime_param_id(id);
    RustcEarlyParamRegion { index: ra_salsa::InternKey::as_intern_id(&interned_id).as_u32() }
}

pub fn to_chalk_trait_id(id: TraitId) -> ChalkTraitId {
    chalk_ir::TraitId(ra_salsa::InternKey::as_intern_id(&id))
}

pub fn from_chalk_trait_id(id: ChalkTraitId) -> TraitId {
    ra_salsa::InternKey::from_intern_id(id.0)
}

pub fn to_rustc_early_binder<T: HasInterner>(
    t: chalk_ir::Binders<T>,
) -> rustc_type_ir::EarlyBinder<RustcInterner, RustcTy> {
    todo!()
}

trait ChalkToRustc<Rustc> {
    fn to_rustc(&self) -> Rustc;
}

impl ChalkToRustc<RustcTy> for chalk_ir::Ty<Interner> {
    fn to_rustc(&self) -> RustcTy {
        match self.kind(Interner) {
            chalk_ir::TyKind::Adt(adt_id, substitution) => {
                let def = RustcAdtDef::new(adt_id.0);
                let args = substitution.to_rustc();
                RustcTy::new(rustc_type_ir::TyKind::Adt(def, args))
            }
            chalk_ir::TyKind::AssociatedType(assoc_type_id, substitution) => {
                let id: TypeAliasId = ra_salsa::InternKey::from_intern_id(assoc_type_id.0);
                let args: RustcGenericArgs = substitution.to_rustc();
                let alias_ty = rustc_type_ir::AliasTy::new(RustcInterner, id.into(), args.iter());
                RustcTy::new(rustc_type_ir::TyKind::Alias(rustc_type_ir::AliasTyKind::Opaque, alias_ty))
            }
            chalk_ir::TyKind::Scalar(scalar) => {
                RustcTy::new(match scalar {
                    chalk_ir::Scalar::Bool => rustc_type_ir::TyKind::Bool,
                    chalk_ir::Scalar::Char => rustc_type_ir::TyKind::Char,
                    chalk_ir::Scalar::Int(chalk_ir::IntTy::Isize) => rustc_type_ir::TyKind::Int(rustc_type_ir::IntTy::Isize),
                    chalk_ir::Scalar::Int(chalk_ir::IntTy::I8) => rustc_type_ir::TyKind::Int(rustc_type_ir::IntTy::I8),
                    chalk_ir::Scalar::Int(chalk_ir::IntTy::I16) => rustc_type_ir::TyKind::Int(rustc_type_ir::IntTy::I16),
                    chalk_ir::Scalar::Int(chalk_ir::IntTy::I32) => rustc_type_ir::TyKind::Int(rustc_type_ir::IntTy::I32),
                    chalk_ir::Scalar::Int(chalk_ir::IntTy::I64) => rustc_type_ir::TyKind::Int(rustc_type_ir::IntTy::I64),
                    chalk_ir::Scalar::Int(chalk_ir::IntTy::I128) => rustc_type_ir::TyKind::Int(rustc_type_ir::IntTy::I128),
                    chalk_ir::Scalar::Uint(chalk_ir::UintTy::Usize) => rustc_type_ir::TyKind::Uint(rustc_type_ir::UintTy::Usize),
                    chalk_ir::Scalar::Uint(chalk_ir::UintTy::U8) => rustc_type_ir::TyKind::Uint(rustc_type_ir::UintTy::U8),
                    chalk_ir::Scalar::Uint(chalk_ir::UintTy::U16) => rustc_type_ir::TyKind::Uint(rustc_type_ir::UintTy::U16),
                    chalk_ir::Scalar::Uint(chalk_ir::UintTy::U32) => rustc_type_ir::TyKind::Uint(rustc_type_ir::UintTy::U32),
                    chalk_ir::Scalar::Uint(chalk_ir::UintTy::U64) => rustc_type_ir::TyKind::Uint(rustc_type_ir::UintTy::U64),
                    chalk_ir::Scalar::Uint(chalk_ir::UintTy::U128) => rustc_type_ir::TyKind::Uint(rustc_type_ir::UintTy::U128),
                    chalk_ir::Scalar::Float(chalk_ir::FloatTy::F16) => rustc_type_ir::TyKind::Float(rustc_type_ir::FloatTy::F16),
                    chalk_ir::Scalar::Float(chalk_ir::FloatTy::F32) => rustc_type_ir::TyKind::Float(rustc_type_ir::FloatTy::F32),
                    chalk_ir::Scalar::Float(chalk_ir::FloatTy::F64) => rustc_type_ir::TyKind::Float(rustc_type_ir::FloatTy::F64),
                    chalk_ir::Scalar::Float(chalk_ir::FloatTy::F128) => rustc_type_ir::TyKind::Float(rustc_type_ir::FloatTy::F128),
                })
            }
            chalk_ir::TyKind::Tuple(_, substitution) => {
                let args = substitution.to_rustc();
                RustcTy::new(rustc_type_ir::TyKind::Tuple(args))
            }
            chalk_ir::TyKind::Array(ty, _) => todo!(),
            chalk_ir::TyKind::Slice(ty) => todo!(),
            chalk_ir::TyKind::Raw(mutability, ty) => todo!(),
            chalk_ir::TyKind::Ref(mutability, lifetime, ty) => todo!(),
            chalk_ir::TyKind::OpaqueType(opaque_ty_id, substitution) => todo!(),
            chalk_ir::TyKind::FnDef(fn_def_id, substitution) => todo!(),
            chalk_ir::TyKind::Str => todo!(),
            chalk_ir::TyKind::Never => todo!(),
            chalk_ir::TyKind::Closure(closure_id, substitution) => todo!(),
            chalk_ir::TyKind::Coroutine(coroutine_id, substitution) => todo!(),
            chalk_ir::TyKind::CoroutineWitness(coroutine_id, substitution) => todo!(),
            chalk_ir::TyKind::Foreign(foreign_def_id) => todo!(),
            chalk_ir::TyKind::Error => todo!(),
            chalk_ir::TyKind::Placeholder(placeholder_index) => todo!(),
            chalk_ir::TyKind::Dyn(dyn_ty) => todo!(),
            chalk_ir::TyKind::Alias(alias_ty) => todo!(),
            chalk_ir::TyKind::Function(fn_pointer) => todo!(),
            chalk_ir::TyKind::BoundVar(bound_var) => todo!(),
            chalk_ir::TyKind::InferenceVar(inference_var, ty_variable_kind) => todo!(),
        }
    }
}

impl ChalkToRustc<RustcRegion> for chalk_ir::Lifetime<Interner> {
    fn to_rustc(&self) -> RustcRegion {
        RustcRegion::new(match self.data(Interner) {
            chalk_ir::LifetimeData::BoundVar(bound_var) => {
                rustc_type_ir::RegionKind::ReBound(
                    bound_var.debruijn.to_rustc(),
                    RustcBoundRegion::new(rustc_type_ir::BoundVar::from_u32(bound_var.index as u32)),
                )
            }
            chalk_ir::LifetimeData::InferenceVar(inference_var) => {
                rustc_type_ir::RegionKind::ReVar(rustc_type_ir::RegionVid::from_u32(inference_var.index()))
            }
            chalk_ir::LifetimeData::Placeholder(placeholder_index) => {
                rustc_type_ir::RegionKind::RePlaceholder(RustcPlaceholderRegion::new(rustc_type_ir::UniverseIndex::from_u32(placeholder_index.ui.counter as u32), rustc_type_ir::BoundVar::from_u32(placeholder_index.idx as u32)))
            }
            chalk_ir::LifetimeData::Static => {
                rustc_type_ir::RegionKind::ReStatic
            }
            chalk_ir::LifetimeData::Erased => {
                rustc_type_ir::RegionKind::ReErased
            }
            chalk_ir::LifetimeData::Phantom(_, _) => {
                unreachable!()
            }
            chalk_ir::LifetimeData::Error => {
                rustc_type_ir::RegionKind::ReError(RustcErrorGuaranteed)
            }
        })
    }
}

impl ChalkToRustc<RustcGenericArgs> for chalk_ir::Substitution<Interner> {
    fn to_rustc(&self) -> RustcGenericArgs {
        RustcGenericArgs::new(self.iter(Interner).map(|arg| -> RustcGenericArg {
            match arg.data(Interner) {
                chalk_ir::GenericArgData::Ty(ty) => ty.to_rustc().into(),
                chalk_ir::GenericArgData::Lifetime(lifetime) => lifetime.to_rustc().into(),
                chalk_ir::GenericArgData::Const(_) => todo!(),
            }
        }))
    }
}

impl ChalkToRustc<RustcTys> for chalk_ir::Substitution<Interner> {
    fn to_rustc(&self) -> RustcTys {
        RustcTys::new(self.iter(Interner).map(|arg| -> RustcTy {
            match arg.data(Interner) {
                chalk_ir::GenericArgData::Ty(ty) => ty.to_rustc(),
                chalk_ir::GenericArgData::Lifetime(lifetime) => panic!(),
                chalk_ir::GenericArgData::Const(_) => panic!(),
            }
        }))
    }
}

impl ChalkToRustc<rustc_type_ir::DebruijnIndex> for chalk_ir::DebruijnIndex {
    fn to_rustc(&self) -> rustc_type_ir::DebruijnIndex {
        rustc_type_ir::DebruijnIndex::from_u32(self.depth())
    }
}

impl ChalkToRustc<rustc_type_ir::UniverseIndex> for chalk_ir::UniverseIndex {
    fn to_rustc(&self) -> rustc_type_ir::UniverseIndex {
        rustc_type_ir::UniverseIndex::from_u32(self.counter as u32)
    }
}
