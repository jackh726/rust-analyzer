#![allow(unused)]

use base_db::CrateId;
use chalk_ir::{ProgramClauseImplication, SeparatorTraitRef};
use hir_def::{hir::PatId, AdtId, BlockId, GenericDefId, TypeAliasId, VariantId};
use intern::{impl_internable, Interned};
use rustc_abi::ReprOptions;
use smallvec::{smallvec, SmallVec};
use std::fmt;
use triomphe::Arc;

use rustc_ast_ir::visit::VisitorResult;
use rustc_index_in_tree::{bit_set::BitSet, IndexVec};
use rustc_type_ir::visit::TypeVisitableExt;
use rustc_type_ir::{
    elaborate, fold,
    inherent::{self, Const as _, Region as _, Ty as _},
    ir_print, relate,
    solve::Reveal,
    visit, BoundVar, CollectAndApply, DebruijnIndex, GenericArgKind, RegionKind, RustIr, TermKind,
    UniverseIndex, Variance, WithCachedTypeInfo,
};

use crate::{
    db::HirDatabase, generics::generics, interner::InternedWrapper, ConstScalar, FnAbi, Interner,
};

use super::{
    abi::Safety,
    fold::{BoundVarReplacer, BoundVarReplacerDelegate, FnMutDelegate},
    generics::Generics,
    mapping::{convert_binder_to_early_binder, ChalkToNextSolver},
    region::{
        BoundRegion, BoundRegionKind, EarlyParamRegion, LateParamRegion, PlaceholderRegion, Region,
    },
    Binder, BoundExistentialPredicate, BoundExistentialPredicates, BoundTy, BoundTyKind,
    CanonicalVarInfo, Clause, Clauses, Const, ConstKind, DefiningOpaqueTypes, ErrorGuaranteed,
    ExprConst, ExternalConstraints, ExternalConstraintsData, GenericArg, GenericArgs,
    InternedClausesWrapper, ParamConst, ParamEnv, ParamTy, PlaceholderConst, PlaceholderTy,
    PredefinedOpaques, PredefinedOpaquesData, Predicate, PredicateKind, Term, Ty, TyKind, Tys,
    ValueConst,
};

impl_internable!(
    InternedWrapper<WithCachedTypeInfo<ConstKind>>,
    InternedWrapper<RegionKind<DbInterner>>,
    InternedWrapper<WithCachedTypeInfo<TyKind>>,
    InternedWrapper<WithCachedTypeInfo<Binder<PredicateKind>>>,
    InternedWrapper<SmallVec<[BoundExistentialPredicate; 2]>>,
    InternedWrapper<SmallVec<[BoundVarKind; 2]>>,
    InternedClausesWrapper,
    InternedWrapper<SmallVec<[GenericArg; 2]>>,
    InternedWrapper<SmallVec<[Ty; 2]>>,
    InternedWrapper<SmallVec<[CanonicalVarInfo; 2]>>,
    InternedWrapper<SmallVec<[GenericDefId; 2]>>,
    InternedWrapper<SmallVec<[Variance; 2]>>,
    InternedWrapper<PredefinedOpaquesData>,
    InternedWrapper<ExternalConstraintsData>,
);

#[macro_export]
#[doc(hidden)]
macro_rules! _interned_vec {
    ($name:ident, $ty:ty) => {
        paste::paste! {
            interned_vec!($name, $ty, nofold);

            impl rustc_type_ir::fold::TypeFoldable<DbInterner> for $name {
                fn try_fold_with<F: rustc_type_ir::fold::FallibleTypeFolder<DbInterner>>(self, folder: &mut F) -> Result<Self, F::Error> {
                    use rustc_type_ir::inherent::{SliceLike as _};
                    Ok($name(intern::Interned::new(InternedWrapper(self.iter().map(|v| v.try_fold_with(folder)).collect::<Result<_, _>>()?))))
                }
            }

            impl rustc_type_ir::visit::TypeVisitable<DbInterner> for $name {
                fn visit_with<V: rustc_type_ir::visit::TypeVisitor<DbInterner>>(&self, visitor: &mut V) -> V::Result {
                    use rustc_type_ir::inherent::{SliceLike as _};
                    use rustc_ast_ir::visit::VisitorResult;
                    rustc_ast_ir::walk_visitable_list!(visitor, self.as_slice().iter());
                    V::Result::output()
                }
            }
        }
    };
    ($name:ident, $ty:ty, nofold) => {
        paste::paste! {
            type [<Interned $name>] = intern::Interned<InternedWrapper<smallvec::SmallVec<[$ty; 2]>>>;

            #[derive(Debug, Clone, PartialEq, Eq, Hash)]
            pub struct $name([<Interned $name>]);

            impl $name {
                pub fn new_from_iter(data: impl IntoIterator<Item = $ty>) -> Self {
                    $name(intern::Interned::new(InternedWrapper(data.into_iter().collect())))
                }
            }

            impl rustc_type_ir::inherent::SliceLike for $name {
                type Item = $ty;

                type IntoIter = <smallvec::SmallVec<[$ty; 2]> as IntoIterator>::IntoIter;

                fn iter(self) -> Self::IntoIter {
                    self.0.0.clone().into_iter()
                }

                fn as_slice(&self) -> &[Self::Item] {
                    self.0.0.as_slice()
                }
            }

            impl IntoIterator for $name {
                type Item = $ty;
                type IntoIter = <Self as rustc_type_ir::inherent::SliceLike>::IntoIter;

                fn into_iter(self) -> Self::IntoIter { rustc_type_ir::inherent::SliceLike::iter(self) }
            }

            impl Default for $name {
                fn default() -> Self {
                    $name(intern::Interned::new(InternedWrapper(Default::default())))
                }
            }
        }
    };
}

pub use crate::_interned_vec as interned_vec;

#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct Symbol;

#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct DbInterner;

impl inherent::DefId<DbInterner> for GenericDefId {
    fn as_local(self) -> Option<GenericDefId> {
        Some(self)
    }
    fn is_local(self) -> bool {
        true
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct Span(Option<span::Span>);

impl inherent::Span<DbInterner> for Span {
    fn dummy() -> Self {
        Span(None)
    }
}

interned_vec!(BoundVarKinds, BoundVarKind, nofold);

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum BoundVarKind {
    Ty(BoundTyKind),
    Region(BoundRegionKind),
    Const,
}

impl BoundVarKind {
    pub fn expect_region(self) -> BoundRegionKind {
        match self {
            BoundVarKind::Region(lt) => lt,
            _ => panic!("expected a region, but found another kind"),
        }
    }

    pub fn expect_ty(self) -> BoundTyKind {
        match self {
            BoundVarKind::Ty(ty) => ty,
            _ => panic!("expected a type, but found another kind"),
        }
    }

    pub fn expect_const(self) {
        match self {
            BoundVarKind::Const => (),
            _ => panic!("expected a const, but found another kind"),
        }
    }
}

interned_vec!(CanonicalVars, CanonicalVarInfo, nofold);

pub struct DepNodeIndex;

#[derive(Debug)]
pub struct Tracked<T: fmt::Debug + Clone>(T);

#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)] // FIXME implement Debug by hand
pub struct Placeholder<T> {
    pub universe: UniverseIndex,
    pub bound: T,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct AllocId;

impl rustc_type_ir::relate::Relate<DbInterner> for PatId {
    fn relate<R: rustc_type_ir::relate::TypeRelation>(
        relation: &mut R,
        a: Self,
        b: Self,
    ) -> rustc_type_ir::relate::RelateResult<DbInterner, Self> {
        // FIXME implement this
        Ok(a)
    }
}

interned_vec!(VariancesOf, Variance, nofold);

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct VariantIdx(VariantId);

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct VariantDef;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct AdtFlags {
    is_enum: bool,
    is_union: bool,
    is_struct: bool,
    has_ctor: bool,
    is_phantom_data: bool,
    is_fundamental: bool,
    is_box: bool,
    is_manually_drop: bool,
    is_variant_list_non_exhaustive: bool,
    is_unsafe_cell: bool,
    is_anonymous: bool,
}

#[derive(Debug, Clone)]
pub struct AdtDefData {
    pub did: GenericDefId,
    pub id: AdtId,
    pub variants: Vec<(VariantIdx, VariantDef)>,
    pub flags: AdtFlags,
    pub repr: ReprOptions,
}

impl PartialEq for AdtDefData {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // There should be only one `AdtDefData` for each `def_id`, therefore
        // it is fine to implement `PartialEq` only based on `def_id`.
        //
        // Below, we exhaustively destructure `self` and `other` so that if the
        // definition of `AdtDefData` changes, a compile-error will be produced,
        // reminding us to revisit this assumption.

        let Self { did: self_def_id, id: _, variants: _, flags: _, repr: _ } = self;
        let Self { did: other_def_id, id: _, variants: _, flags: _, repr: _ } = other;

        let res = self_def_id == other_def_id;

        // Double check that implicit assumption detailed above.
        if cfg!(debug_assertions) && res {
            let deep = self.flags == other.flags
                && self.repr == other.repr
                && self.variants == other.variants;
            assert!(deep, "AdtDefData for the same def-id has differing data");
        }

        res
    }
}

impl Eq for AdtDefData {}

/// There should be only one AdtDef for each `did`, therefore
/// it is fine to implement `Hash` only based on `did`.
impl std::hash::Hash for AdtDefData {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, s: &mut H) {
        self.did.hash(s)
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct AdtDef(AdtDefData);

impl AdtDef {
    pub fn new(def_id: AdtId) -> Self {
        todo!()
    }

    pub fn is_enum(&self) -> bool {
        self.0.flags.is_enum
    }

    #[inline]
    pub fn repr(self) -> ReprOptions {
        self.0.repr
    }
}

impl inherent::AdtDef<DbInterner> for AdtDef {
    fn def_id(&self) -> <DbInterner as rustc_type_ir::Interner>::DefId {
        self.0.did
    }

    fn is_struct(&self) -> bool {
        self.0.flags.is_struct
    }

    fn is_phantom_data(&self) -> bool {
        self.0.flags.is_phantom_data
    }

    fn is_fundamental(&self) -> bool {
        self.0.flags.is_fundamental
    }
}

impl<'cx> inherent::IrAdtDef<DbInterner, DbIr<'cx>> for AdtDef {
    fn struct_tail_ty(
        self,
        ir: DbIr<'cx>,
    ) -> Option<rustc_type_ir::EarlyBinder<DbInterner, <DbInterner as rustc_type_ir::Interner>::Ty>>
    {
        let db = ir.db;
        let hir_def::AdtId::StructId(struct_id) = self.0.id else {
            return None;
        };
        let id: VariantId = struct_id.into();
        let variant_data = &id.variant_data(db.upcast());
        let Some((last_idx, _)) = variant_data.fields().iter().last() else { return None };
        let field_types = db.field_types(id);

        let last_ty: rustc_type_ir::Binder<DbInterner, Ty> =
            field_types[last_idx].clone().to_nextsolver();
        Some(convert_binder_to_early_binder(last_ty))
    }

    fn all_field_tys(
        self,
        ir: DbIr<'cx>,
    ) -> rustc_type_ir::EarlyBinder<
        DbInterner,
        impl IntoIterator<Item = <DbInterner as rustc_type_ir::Interner>::Ty>,
    > {
        let db = ir.db;
        let id = match self.0.id {
            AdtId::StructId(struct_id) => VariantId::StructId(struct_id),
            AdtId::UnionId(union_id) => VariantId::UnionId(union_id),
            AdtId::EnumId(enum_id) => todo!(),
        };
        let variant_data = id.variant_data(db.upcast());
        let field_types = db.field_types(id);
        let fields: Vec<_> = variant_data.fields().iter().map(|(idx, _)| idx).collect();
        let tys = fields.into_iter().map(move |idx| {
            let ty: rustc_type_ir::Binder<DbInterner, Ty> =
                field_types[idx].clone().to_nextsolver();
            let ty = convert_binder_to_early_binder(ty);
            ty.skip_binder()
        });
        rustc_type_ir::EarlyBinder::bind(tys)
    }

    fn sized_constraint(
        self,
        interner: DbIr<'cx>,
    ) -> Option<rustc_type_ir::EarlyBinder<DbInterner, <DbInterner as rustc_type_ir::Interner>::Ty>>
    {
        todo!()
    }

    fn destructor(self, ir: DbIr<'cx>) -> Option<rustc_type_ir::solve::AdtDestructorKind> {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Features;

impl inherent::Features<DbInterner> for Features {
    fn generic_const_exprs(self) -> bool {
        false
    }

    fn coroutine_clone(self) -> bool {
        false
    }

    fn associated_const_equality(self) -> bool {
        false
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct UnsizingParams(BitSet<u32>);

impl std::ops::Deref for UnsizingParams {
    type Target = BitSet<u32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl rustc_type_ir::Interner for DbInterner {
    type DefId = GenericDefId;
    type LocalDefId = GenericDefId;
    type Span = Span;

    type GenericArgs = GenericArgs;
    type GenericArgsSlice = GenericArgs;
    type GenericArg = GenericArg;

    type Term = Term;

    type BoundVarKinds = BoundVarKinds;
    type BoundVarKind = BoundVarKind;

    type PredefinedOpaques = PredefinedOpaques;

    fn mk_predefined_opaques_in_body(
        self,
        data: rustc_type_ir::solve::PredefinedOpaquesData<Self>,
    ) -> Self::PredefinedOpaques {
        PredefinedOpaques::new(data)
    }

    type DefiningOpaqueTypes = DefiningOpaqueTypes;

    type CanonicalVars = CanonicalVars;

    fn mk_canonical_var_infos(
        self,
        infos: &[rustc_type_ir::CanonicalVarInfo<Self>],
    ) -> Self::CanonicalVars {
        CanonicalVars::new_from_iter(infos.iter().cloned())
    }

    type ExternalConstraints = ExternalConstraints;

    fn mk_external_constraints(
        self,
        data: rustc_type_ir::solve::ExternalConstraintsData<Self>,
    ) -> Self::ExternalConstraints {
        ExternalConstraints::new(data)
    }

    type DepNodeIndex = DepNodeIndex;

    type Tracked<T: fmt::Debug + Clone> = Tracked<T>;

    fn mk_tracked<T: fmt::Debug + Clone>(
        self,
        data: T,
        dep_node: Self::DepNodeIndex,
    ) -> Self::Tracked<T> {
        Tracked(data)
    }

    fn get_tracked<T: fmt::Debug + Clone>(self, tracked: &Self::Tracked<T>) -> T {
        tracked.0.clone()
    }

    fn with_cached_task<T>(self, task: impl FnOnce() -> T) -> (T, Self::DepNodeIndex) {
        (task(), DepNodeIndex)
    }

    type Ty = Ty;
    type Tys = Tys;
    type FnInputTys = Tys;
    type ParamTy = ParamTy;
    type BoundTy = BoundTy;
    type PlaceholderTy = PlaceholderTy;

    type ErrorGuaranteed = ErrorGuaranteed;
    type BoundExistentialPredicates = BoundExistentialPredicates;
    type AllocId = AllocId;
    type Pat = PatId;
    type Safety = Safety;
    type Abi = FnAbi;

    type Const = Const;
    type PlaceholderConst = PlaceholderConst;
    type ParamConst = ParamConst;
    type BoundConst = rustc_type_ir::BoundVar;
    type ValueConst = ValueConst;
    type ExprConst = ExprConst;

    type Region = Region;
    type EarlyParamRegion = EarlyParamRegion;
    type LateParamRegion = LateParamRegion;
    type BoundRegion = BoundRegion;
    type PlaceholderRegion = PlaceholderRegion;

    type ParamEnv = ParamEnv;
    type Predicate = Predicate;
    type Clause = Clause;
    type Clauses = Clauses;

    fn with_global_cache<R>(
        self,
        f: impl FnOnce(&mut rustc_type_ir::search_graph::GlobalCache<Self>) -> R,
    ) -> R {
        let mut cache = rustc_type_ir::search_graph::GlobalCache::default();
        f(&mut cache)
    }

    fn evaluation_is_concurrent(&self) -> bool {
        false
    }

    fn expand_abstract_consts<T: rustc_type_ir::fold::TypeFoldable<Self>>(self, t: T) -> T {
        t
    }

    type GenericsOf = Generics;

    fn generics_of(self, def_id: Self::DefId) -> Self::GenericsOf {
        todo!()
    }

    type VariancesOf = VariancesOf;

    fn variances_of(self, def_id: Self::DefId) -> Self::VariancesOf {
        todo!()
    }

    fn type_of(self, def_id: Self::DefId) -> rustc_type_ir::EarlyBinder<Self, Self::Ty> {
        todo!()
    }

    type AdtDef = AdtDef;

    fn adt_def(self, adt_def_id: Self::DefId) -> Self::AdtDef {
        todo!()
    }

    fn alias_ty_kind(self, alias: rustc_type_ir::AliasTy<Self>) -> rustc_type_ir::AliasTyKind {
        todo!()
    }

    fn alias_term_kind(
        self,
        alias: rustc_type_ir::AliasTerm<Self>,
    ) -> rustc_type_ir::AliasTermKind {
        todo!()
    }

    fn trait_ref_and_own_args_for_alias(
        self,
        def_id: Self::DefId,
        args: Self::GenericArgs,
    ) -> (rustc_type_ir::TraitRef<Self>, Self::GenericArgsSlice) {
        todo!()
    }

    fn mk_args(self, args: &[Self::GenericArg]) -> Self::GenericArgs {
        GenericArgs::new_from_iter(args.iter().cloned())
    }

    fn mk_args_from_iter<I, T>(self, args: I) -> T::Output
    where
        I: Iterator<Item = T>,
        T: rustc_type_ir::CollectAndApply<Self::GenericArg, Self::GenericArgs>,
    {
        CollectAndApply::collect_and_apply(args, |g| GenericArgs::new_from_iter(g.iter().cloned()))
    }

    fn check_args_compatible(self, def_id: Self::DefId, args: Self::GenericArgs) -> bool {
        todo!()
    }

    fn debug_assert_args_compatible(self, def_id: Self::DefId, args: Self::GenericArgs) {}

    fn debug_assert_existential_args_compatible(
        self,
        def_id: Self::DefId,
        args: Self::GenericArgs,
    ) {
    }

    fn mk_type_list_from_iter<I, T>(self, args: I) -> T::Output
    where
        I: Iterator<Item = T>,
        T: rustc_type_ir::CollectAndApply<Self::Ty, Self::Tys>,
    {
        CollectAndApply::collect_and_apply(args, |g| Tys::new_from_iter(g.iter().cloned()))
    }

    fn parent(self, def_id: Self::DefId) -> Self::DefId {
        todo!()
    }

    fn recursion_limit(self) -> usize {
        50
    }

    type Features = Features;

    fn features(self) -> Self::Features {
        Features
    }

    fn bound_coroutine_hidden_types(
        self,
        def_id: Self::DefId,
    ) -> impl IntoIterator<Item = rustc_type_ir::EarlyBinder<Self, rustc_type_ir::Binder<Self, Self::Ty>>>
    {
        [todo!()]
    }

    fn fn_sig(
        self,
        def_id: Self::DefId,
    ) -> rustc_type_ir::EarlyBinder<Self, rustc_type_ir::Binder<Self, rustc_type_ir::FnSig<Self>>>
    {
        todo!()
    }

    fn coroutine_movability(self, def_id: Self::DefId) -> rustc_ast_ir::Movability {
        todo!()
    }

    fn coroutine_for_closure(self, def_id: Self::DefId) -> Self::DefId {
        todo!()
    }

    fn generics_require_sized_self(self, def_id: Self::DefId) -> bool {
        todo!()
    }

    fn item_bounds(
        self,
        def_id: Self::DefId,
    ) -> rustc_type_ir::EarlyBinder<Self, impl IntoIterator<Item = Self::Clause>> {
        rustc_type_ir::EarlyBinder::bind([todo!()])
    }

    fn predicates_of(
        self,
        def_id: Self::DefId,
    ) -> rustc_type_ir::EarlyBinder<Self, impl IntoIterator<Item = Self::Clause>> {
        rustc_type_ir::EarlyBinder::bind([todo!()])
    }

    fn own_predicates_of(
        self,
        def_id: Self::DefId,
    ) -> rustc_type_ir::EarlyBinder<Self, impl IntoIterator<Item = Self::Clause>> {
        rustc_type_ir::EarlyBinder::bind([todo!()])
    }

    fn explicit_super_predicates_of(
        self,
        def_id: Self::DefId,
    ) -> rustc_type_ir::EarlyBinder<Self, impl IntoIterator<Item = (Self::Clause, Self::Span)>>
    {
        rustc_type_ir::EarlyBinder::bind([todo!()])
    }

    fn explicit_implied_predicates_of(
        self,
        def_id: Self::DefId,
    ) -> rustc_type_ir::EarlyBinder<Self, impl IntoIterator<Item = (Self::Clause, Self::Span)>>
    {
        rustc_type_ir::EarlyBinder::bind([todo!()])
    }

    fn const_conditions(
        self,
        def_id: Self::DefId,
    ) -> rustc_type_ir::EarlyBinder<
        Self,
        impl IntoIterator<Item = rustc_type_ir::Binder<Self, rustc_type_ir::TraitRef<Self>>>,
    > {
        rustc_type_ir::EarlyBinder::bind([todo!()])
    }

    fn has_target_features(self, def_id: Self::DefId) -> bool {
        todo!()
    }

    fn require_lang_item(
        self,
        lang_item: rustc_type_ir::lang_items::TraitSolverLangItem,
    ) -> Self::DefId {
        todo!()
    }

    fn is_lang_item(
        self,
        def_id: Self::DefId,
        lang_item: rustc_type_ir::lang_items::TraitSolverLangItem,
    ) -> bool {
        todo!()
    }

    fn as_lang_item(
        self,
        def_id: Self::DefId,
    ) -> Option<rustc_type_ir::lang_items::TraitSolverLangItem> {
        todo!()
    }

    fn associated_type_def_ids(self, def_id: Self::DefId) -> impl IntoIterator<Item = Self::DefId> {
        [todo!()]
    }

    fn for_each_relevant_impl(
        self,
        trait_def_id: Self::DefId,
        self_ty: Self::Ty,
        f: impl FnMut(Self::DefId),
    ) {
        todo!()
    }

    fn has_item_definition(self, def_id: Self::DefId) -> bool {
        todo!()
    }

    fn impl_is_default(self, impl_def_id: Self::DefId) -> bool {
        todo!()
    }

    fn impl_trait_ref(
        self,
        impl_def_id: Self::DefId,
    ) -> rustc_type_ir::EarlyBinder<Self, rustc_type_ir::TraitRef<Self>> {
        todo!()
    }

    fn impl_polarity(self, impl_def_id: Self::DefId) -> rustc_type_ir::ImplPolarity {
        todo!()
    }

    fn trait_is_auto(self, trait_def_id: Self::DefId) -> bool {
        todo!()
    }

    fn trait_is_alias(self, trait_def_id: Self::DefId) -> bool {
        todo!()
    }

    fn trait_is_dyn_compatible(self, trait_def_id: Self::DefId) -> bool {
        todo!()
    }

    fn trait_is_fundamental(self, def_id: Self::DefId) -> bool {
        todo!()
    }

    fn trait_may_be_implemented_via_object(self, trait_def_id: Self::DefId) -> bool {
        todo!()
    }

    fn is_impl_trait_in_trait(self, def_id: Self::DefId) -> bool {
        todo!()
    }

    fn delay_bug(self, msg: impl ToString) -> Self::ErrorGuaranteed {
        todo!()
    }

    fn is_general_coroutine(self, coroutine_def_id: Self::DefId) -> bool {
        todo!()
    }

    fn coroutine_is_async(self, coroutine_def_id: Self::DefId) -> bool {
        todo!()
    }

    fn coroutine_is_gen(self, coroutine_def_id: Self::DefId) -> bool {
        todo!()
    }

    fn coroutine_is_async_gen(self, coroutine_def_id: Self::DefId) -> bool {
        todo!()
    }

    type UnsizingParams = UnsizingParams;

    fn unsizing_params_for_adt(self, adt_def_id: Self::DefId) -> Self::UnsizingParams {
        todo!()
    }

    fn find_const_ty_from_env(
        self,
        param_env: &Self::ParamEnv,
        placeholder: Self::PlaceholderConst,
    ) -> Self::Ty {
        todo!()
    }

    fn anonymize_bound_vars<T: rustc_type_ir::fold::TypeFoldable<Self>>(
        self,
        binder: rustc_type_ir::Binder<Self, T>,
    ) -> rustc_type_ir::Binder<Self, T> {
        todo!()
    }

    fn opaque_types_defined_by(
        self,
        defining_anchor: Self::LocalDefId,
    ) -> Self::DefiningOpaqueTypes {
        todo!()
    }

    fn alias_has_const_conditions(self, def_id: Self::DefId) -> bool {
        todo!()
    }

    fn explicit_implied_const_bounds(
        self,
        def_id: Self::DefId,
    ) -> rustc_type_ir::EarlyBinder<
        Self,
        impl IntoIterator<Item = rustc_type_ir::Binder<Self, rustc_type_ir::TraitRef<Self>>>,
    > {
        rustc_type_ir::EarlyBinder::bind([todo!()])
    }

    fn fn_is_const(self, def_id: Self::DefId) -> bool {
        todo!()
    }

    fn impl_is_const(self, def_id: Self::DefId) -> bool {
        todo!()
    }
}

impl DbInterner {
    pub fn shift_bound_var_indices<T>(self, bound_vars: usize, value: T) -> T
    where
        T: rustc_type_ir::fold::TypeFoldable<Self>,
    {
        let shift_bv = |bv: BoundVar| BoundVar::from_usize(bv.as_usize() + bound_vars);
        self.replace_escaping_bound_vars_uncached(
            value,
            FnMutDelegate {
                regions: &mut |r: BoundRegion| {
                    Region::new_bound(
                        self,
                        DebruijnIndex::ZERO,
                        BoundRegion { var: shift_bv(r.var), kind: r.kind },
                    )
                },
                types: &mut |t: BoundTy| {
                    Ty::new_bound(
                        self,
                        DebruijnIndex::ZERO,
                        BoundTy { var: shift_bv(t.var), kind: t.kind },
                    )
                },
                consts: &mut |c| Const::new_bound(self, DebruijnIndex::ZERO, shift_bv(c)),
            },
        )
    }

    pub fn replace_escaping_bound_vars_uncached<
        T: rustc_type_ir::fold::TypeFoldable<DbInterner>,
    >(
        self,
        value: T,
        delegate: impl BoundVarReplacerDelegate,
    ) -> T {
        if !value.has_escaping_bound_vars() {
            value
        } else {
            let mut replacer = BoundVarReplacer::new(self, delegate);
            value.fold_with(&mut replacer)
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct DbIr<'a> {
    pub(crate) db: &'a dyn HirDatabase,
}

impl<'a> DbIr<'a> {
    pub fn new(db: &'a dyn HirDatabase) -> Self {
        DbIr { db }
    }
}
impl<'cx> RustIr for DbIr<'cx> {
    type Interner = DbInterner;

    fn interner(self) -> Self::Interner {
        DbInterner
    }
}

macro_rules! TrivialTypeTraversalImpls {
    ($($ty:ty,)+) => {
        $(
            impl rustc_type_ir::fold::TypeFoldable<DbInterner> for $ty {
                fn try_fold_with<F: rustc_type_ir::fold::FallibleTypeFolder<DbInterner>>(
                    self,
                    _: &mut F,
                ) -> ::std::result::Result<Self, F::Error> {
                    Ok(self)
                }

                #[inline]
                fn fold_with<F: rustc_type_ir::fold::TypeFolder<DbInterner>>(
                    self,
                    _: &mut F,
                ) -> Self {
                    self
                }
            }

            impl rustc_type_ir::visit::TypeVisitable<DbInterner> for $ty {
                #[inline]
                fn visit_with<F: rustc_type_ir::visit::TypeVisitor<DbInterner>>(
                    &self,
                    _: &mut F)
                    -> F::Result
                {
                    <F::Result as rustc_ast_ir::visit::VisitorResult>::output()
                }
            }
        )+
    };
}

TrivialTypeTraversalImpls! {
    GenericDefId,
    PatId,
    Safety,
    FnAbi,
    Span,
    ParamConst,
    ParamTy,
    BoundRegion,
    BoundVar,
    Placeholder<BoundRegion>,
    Placeholder<BoundTy>,
    Placeholder<BoundVar>,
    ValueConst,
    Reveal,
}
