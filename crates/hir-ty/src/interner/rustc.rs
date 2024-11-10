#![allow(unused)]

use base_db::ra_salsa::InternId;
use chalk_ir::{ProgramClauseImplication, SeparatorTraitRef, Variance};
use hir_def::TypeAliasId;
use intern::{impl_internable, Interned};
use smallvec::{smallvec, SmallVec};
use span::Span;
use std::fmt;
use triomphe::Arc;

use rustc_ast_ir::visit::VisitorResult;
use rustc_index_in_tree::bit_set::BitSet;
use rustc_type_ir::{
    elaborate, fold, inherent, ir_print, relate,
    solve::{ExternalConstraintsData, PredefinedOpaquesData},
    visit, CanonicalVarInfo, ConstKind, GenericArgKind, TermKind,
};

use super::InternedWrapper;

impl_internable!(InternedWrapper<SmallVec<[RustcGenericArg; 2]>>,);

#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct RustcInterner;

macro_rules! todo_structural {
    ($t:ty) => {
        impl relate::Relate<RustcInterner> for $t {
            fn relate<R: relate::TypeRelation<RustcInterner>>(
                _relation: &mut R,
                _a: Self,
                _b: Self,
            ) -> relate::RelateResult<RustcInterner, Self> {
                todo!()
            }
        }

        impl fold::TypeFoldable<RustcInterner> for $t {
            fn try_fold_with<F: fold::FallibleTypeFolder<RustcInterner>>(
                self,
                _folder: &mut F,
            ) -> Result<Self, F::Error> {
                todo!()
            }
        }

        impl visit::TypeVisitable<RustcInterner> for $t {
            fn visit_with<V: visit::TypeVisitor<RustcInterner>>(
                &self,
                _visitor: &mut V,
            ) -> V::Result {
                todo!()
            }
        }
    };
}

impl inherent::DefId<RustcInterner> for InternId {
    fn as_local(self) -> Option<InternId> {
        Some(self)
    }
    fn is_local(self) -> bool {
        true
    }
}

todo_structural!(InternId);

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct RustcSpan(Option<Span>);

todo_structural!(RustcSpan);

impl inherent::Span<RustcInterner> for RustcSpan {
    fn dummy() -> Self {
        RustcSpan(None)
    }
}

type InternedGenericArgs = Interned<InternedWrapper<SmallVec<[RustcGenericArg; 2]>>>;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct GenericArgs(InternedGenericArgs);

impl inherent::GenericArgs<RustcInterner> for GenericArgs {
    fn dummy() -> Self {
        GenericArgs(Interned::new(InternedWrapper(smallvec![])))
    }

    fn rebase_onto(
        self,
        interner: RustcInterner,
        source_def_id: <RustcInterner as rustc_type_ir::Interner>::DefId,
        target: <RustcInterner as rustc_type_ir::Interner>::GenericArgs,
    ) -> <RustcInterner as rustc_type_ir::Interner>::GenericArgs {
        todo!()
    }

    fn type_at(self, i: usize) -> <RustcInterner as rustc_type_ir::Interner>::Ty {
        todo!()
    }

    fn region_at(self, i: usize) -> <RustcInterner as rustc_type_ir::Interner>::Region {
        todo!()
    }

    fn const_at(self, i: usize) -> <RustcInterner as rustc_type_ir::Interner>::Const {
        todo!()
    }

    fn identity_for_item(
        interner: RustcInterner,
        def_id: <RustcInterner as rustc_type_ir::Interner>::DefId,
    ) -> <RustcInterner as rustc_type_ir::Interner>::GenericArgs {
        todo!()
    }

    fn extend_with_error(
        interner: RustcInterner,
        def_id: <RustcInterner as rustc_type_ir::Interner>::DefId,
        original_args: &[RustcGenericArg],
    ) -> <RustcInterner as rustc_type_ir::Interner>::GenericArgs {
        todo!()
    }

    fn split_closure_args(self) -> rustc_type_ir::ClosureArgsParts<RustcInterner> {
        todo!()
    }

    fn split_coroutine_closure_args(
        self,
    ) -> rustc_type_ir::CoroutineClosureArgsParts<RustcInterner> {
        todo!()
    }

    fn split_coroutine_args(self) -> rustc_type_ir::CoroutineArgsParts<RustcInterner> {
        todo!()
    }
}

todo_structural!(GenericArgs);

pub struct GenericArgsIter;
impl Iterator for GenericArgsIter {
    type Item = RustcGenericArg;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl inherent::SliceLike for GenericArgs {
    type Item = RustcGenericArg;
    type IntoIter = GenericArgsIter;

    fn iter(self) -> Self::IntoIter {
        todo!()
    }

    fn as_slice(&self) -> &[Self::Item] {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RustcGenericArg;

todo_structural!(RustcGenericArg);

impl inherent::GenericArg<RustcInterner> for RustcGenericArg {}

impl inherent::IntoKind for RustcGenericArg {
    type Kind = GenericArgKind<RustcInterner>;

    fn kind(self) -> Self::Kind {
        todo!()
    }
}

impl From<RustcTy> for RustcGenericArg {
    fn from(value: RustcTy) -> Self {
        todo!()
    }
}

impl From<RustcConst> for RustcGenericArg {
    fn from(value: RustcConst) -> Self {
        todo!()
    }
}

impl From<RustcRegion> for RustcGenericArg {
    fn from(value: RustcRegion) -> Self {
        todo!()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RustcTy;

todo_structural!(RustcTy);

impl inherent::Ty<RustcInterner> for RustcTy {
    fn new_unit(interner: RustcInterner) -> Self {
        todo!()
    }

    fn new_bool(interner: RustcInterner) -> Self {
        todo!()
    }

    fn new_u8(interner: RustcInterner) -> Self {
        todo!()
    }

    fn new_usize(interner: RustcInterner) -> Self {
        todo!()
    }

    fn new_infer(interner: RustcInterner, var: rustc_type_ir::InferTy) -> Self {
        todo!()
    }

    fn new_var(interner: RustcInterner, var: rustc_type_ir::TyVid) -> Self {
        todo!()
    }

    fn new_param(
        interner: RustcInterner,
        param: <RustcInterner as rustc_type_ir::Interner>::ParamTy,
    ) -> Self {
        todo!()
    }

    fn new_placeholder(
        interner: RustcInterner,
        param: <RustcInterner as rustc_type_ir::Interner>::PlaceholderTy,
    ) -> Self {
        todo!()
    }

    fn new_bound(
        interner: RustcInterner,
        debruijn: rustc_type_ir::DebruijnIndex,
        var: <RustcInterner as rustc_type_ir::Interner>::BoundTy,
    ) -> Self {
        todo!()
    }

    fn new_anon_bound(
        interner: RustcInterner,
        debruijn: rustc_type_ir::DebruijnIndex,
        var: rustc_type_ir::BoundVar,
    ) -> Self {
        todo!()
    }

    fn new_alias(
        interner: RustcInterner,
        kind: rustc_type_ir::AliasTyKind,
        alias_ty: rustc_type_ir::AliasTy<RustcInterner>,
    ) -> Self {
        todo!()
    }

    fn new_error(
        interner: RustcInterner,
        guar: <RustcInterner as rustc_type_ir::Interner>::ErrorGuaranteed,
    ) -> Self {
        todo!()
    }

    fn new_adt(
        interner: RustcInterner,
        adt_def: <RustcInterner as rustc_type_ir::Interner>::AdtDef,
        args: <RustcInterner as rustc_type_ir::Interner>::GenericArgs,
    ) -> Self {
        todo!()
    }

    fn new_foreign(
        interner: RustcInterner,
        def_id: <RustcInterner as rustc_type_ir::Interner>::DefId,
    ) -> Self {
        todo!()
    }

    fn new_dynamic(
        interner: RustcInterner,
        preds: <RustcInterner as rustc_type_ir::Interner>::BoundExistentialPredicates,
        region: <RustcInterner as rustc_type_ir::Interner>::Region,
        kind: rustc_type_ir::DynKind,
    ) -> Self {
        todo!()
    }

    fn new_coroutine(
        interner: RustcInterner,
        def_id: <RustcInterner as rustc_type_ir::Interner>::DefId,
        args: <RustcInterner as rustc_type_ir::Interner>::GenericArgs,
    ) -> Self {
        todo!()
    }

    fn new_coroutine_closure(
        interner: RustcInterner,
        def_id: <RustcInterner as rustc_type_ir::Interner>::DefId,
        args: <RustcInterner as rustc_type_ir::Interner>::GenericArgs,
    ) -> Self {
        todo!()
    }

    fn new_closure(
        interner: RustcInterner,
        def_id: <RustcInterner as rustc_type_ir::Interner>::DefId,
        args: <RustcInterner as rustc_type_ir::Interner>::GenericArgs,
    ) -> Self {
        todo!()
    }

    fn new_coroutine_witness(
        interner: RustcInterner,
        def_id: <RustcInterner as rustc_type_ir::Interner>::DefId,
        args: <RustcInterner as rustc_type_ir::Interner>::GenericArgs,
    ) -> Self {
        todo!()
    }

    fn new_ptr(interner: RustcInterner, ty: Self, mutbl: rustc_ast_ir::Mutability) -> Self {
        todo!()
    }

    fn new_ref(
        interner: RustcInterner,
        region: <RustcInterner as rustc_type_ir::Interner>::Region,
        ty: Self,
        mutbl: rustc_ast_ir::Mutability,
    ) -> Self {
        todo!()
    }

    fn new_array_with_const_len(
        interner: RustcInterner,
        ty: Self,
        len: <RustcInterner as rustc_type_ir::Interner>::Const,
    ) -> Self {
        todo!()
    }

    fn new_slice(interner: RustcInterner, ty: Self) -> Self {
        todo!()
    }

    fn new_tup(
        interner: RustcInterner,
        tys: &[<RustcInterner as rustc_type_ir::Interner>::Ty],
    ) -> Self {
        todo!()
    }

    fn new_tup_from_iter<It, T>(interner: RustcInterner, iter: It) -> T::Output
    where
        It: Iterator<Item = T>,
        T: rustc_type_ir::CollectAndApply<Self, Self>,
    {
        todo!()
    }

    fn new_fn_def(
        interner: RustcInterner,
        def_id: <RustcInterner as rustc_type_ir::Interner>::DefId,
        args: <RustcInterner as rustc_type_ir::Interner>::GenericArgs,
    ) -> Self {
        todo!()
    }

    fn new_fn_ptr(
        interner: RustcInterner,
        sig: rustc_type_ir::Binder<RustcInterner, rustc_type_ir::FnSig<RustcInterner>>,
    ) -> Self {
        todo!()
    }

    fn new_pat(
        interner: RustcInterner,
        ty: Self,
        pat: <RustcInterner as rustc_type_ir::Interner>::Pat,
    ) -> Self {
        todo!()
    }

    fn tuple_fields(self) -> <RustcInterner as rustc_type_ir::Interner>::Tys {
        todo!()
    }

    fn to_opt_closure_kind(self) -> Option<rustc_type_ir::ClosureKind> {
        todo!()
    }

    fn from_closure_kind(interner: RustcInterner, kind: rustc_type_ir::ClosureKind) -> Self {
        todo!()
    }

    fn from_coroutine_closure_kind(
        interner: RustcInterner,
        kind: rustc_type_ir::ClosureKind,
    ) -> Self {
        todo!()
    }

    fn discriminant_ty(
        self,
        interner: RustcInterner,
    ) -> <RustcInterner as rustc_type_ir::Interner>::Ty {
        todo!()
    }

    fn async_destructor_ty(
        self,
        interner: RustcInterner,
    ) -> <RustcInterner as rustc_type_ir::Interner>::Ty {
        todo!()
    }
}

impl visit::Flags for RustcTy {
    fn flags(&self) -> rustc_type_ir::TypeFlags {
        todo!()
    }

    fn outer_exclusive_binder(&self) -> rustc_type_ir::DebruijnIndex {
        todo!()
    }
}

impl fold::TypeSuperFoldable<RustcInterner> for RustcTy {
    fn try_super_fold_with<F: fold::FallibleTypeFolder<RustcInterner>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        todo!()
    }
}

impl visit::TypeSuperVisitable<RustcInterner> for RustcTy {
    fn super_visit_with<V: visit::TypeVisitor<RustcInterner>>(&self, visitor: &mut V) -> V::Result {
        todo!()
    }
}

impl inherent::IntoKind for RustcTy {
    type Kind = rustc_type_ir::TyKind<RustcInterner>;

    fn kind(self) -> Self::Kind {
        todo!()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RustcConst;

todo_structural!(RustcConst);

impl inherent::Const<RustcInterner> for RustcConst {
    fn try_to_target_usize(self, interner: RustcInterner) -> Option<u64> {
        todo!()
    }

    fn new_infer(interner: RustcInterner, var: rustc_type_ir::InferConst) -> Self {
        todo!()
    }

    fn new_var(interner: RustcInterner, var: rustc_type_ir::ConstVid) -> Self {
        todo!()
    }

    fn new_bound(
        interner: RustcInterner,
        debruijn: rustc_type_ir::DebruijnIndex,
        var: <RustcInterner as rustc_type_ir::Interner>::BoundConst,
    ) -> Self {
        todo!()
    }

    fn new_anon_bound(
        interner: RustcInterner,
        debruijn: rustc_type_ir::DebruijnIndex,
        var: rustc_type_ir::BoundVar,
    ) -> Self {
        todo!()
    }

    fn new_unevaluated(
        interner: RustcInterner,
        uv: rustc_type_ir::UnevaluatedConst<RustcInterner>,
    ) -> Self {
        todo!()
    }

    fn new_expr(
        interner: RustcInterner,
        expr: <RustcInterner as rustc_type_ir::Interner>::ExprConst,
    ) -> Self {
        todo!()
    }

    fn new_error(
        interner: RustcInterner,
        guar: <RustcInterner as rustc_type_ir::Interner>::ErrorGuaranteed,
    ) -> Self {
        todo!()
    }
}

impl visit::Flags for RustcConst {
    fn flags(&self) -> rustc_type_ir::TypeFlags {
        todo!()
    }

    fn outer_exclusive_binder(&self) -> rustc_type_ir::DebruijnIndex {
        todo!()
    }
}

impl fold::TypeSuperFoldable<RustcInterner> for RustcConst {
    fn try_super_fold_with<F: fold::FallibleTypeFolder<RustcInterner>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        todo!()
    }
}

impl visit::TypeSuperVisitable<RustcInterner> for RustcConst {
    fn super_visit_with<V: visit::TypeVisitor<RustcInterner>>(&self, visitor: &mut V) -> V::Result {
        todo!()
    }
}

impl inherent::IntoKind for RustcConst {
    type Kind = ConstKind<RustcInterner>;

    fn kind(self) -> Self::Kind {
        todo!()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RustcTerm {
    Ty(RustcTy),
    Const(RustcConst),
}

impl inherent::Term<RustcInterner> for RustcTerm {}

todo_structural!(RustcTerm);

impl inherent::IntoKind for RustcTerm {
    type Kind = TermKind<RustcInterner>;

    fn kind(self) -> Self::Kind {
        match self {
            Self::Ty(ty) => TermKind::Ty(ty),
            Self::Const(ct) => TermKind::Const(ct),
        }
    }
}

impl From<RustcTy> for RustcTerm {
    fn from(value: RustcTy) -> Self {
        todo!()
    }
}

impl From<RustcConst> for RustcTerm {
    fn from(value: RustcConst) -> Self {
        todo!()
    }
}

impl<T> ir_print::IrPrint<T> for RustcInterner {
    fn print(t: &T, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
    }

    fn print_debug(t: &T, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RustcBoundVarKinds;

todo_structural!(RustcBoundVarKinds);

impl Default for RustcBoundVarKinds {
    fn default() -> Self {
        todo!()
    }
}

pub struct BoundVarKindsIter;
impl Iterator for BoundVarKindsIter {
    type Item = RustcBoundVarKind;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl inherent::SliceLike for RustcBoundVarKinds {
    type Item = RustcBoundVarKind;
    type IntoIter = BoundVarKindsIter;

    fn iter(self) -> Self::IntoIter {
        todo!()
    }

    fn as_slice(&self) -> &[Self::Item] {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RustcBoundVarKind;

todo_structural!(RustcBoundVarKind);

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RustcPredefinedOpaques;

todo_structural!(RustcPredefinedOpaques);

impl std::ops::Deref for RustcPredefinedOpaques {
    type Target = PredefinedOpaquesData<RustcInterner>;

    fn deref(&self) -> &Self::Target {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RustcDefiningOpaqueTypes;

todo_structural!(RustcDefiningOpaqueTypes);

impl Default for RustcDefiningOpaqueTypes {
    fn default() -> Self {
        todo!()
    }
}

pub struct RustcDefiningOpaqueTypesIter;
impl Iterator for RustcDefiningOpaqueTypesIter {
    type Item = InternId;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl inherent::SliceLike for RustcDefiningOpaqueTypes {
    type Item = InternId;
    type IntoIter = RustcDefiningOpaqueTypesIter;

    fn iter(self) -> Self::IntoIter {
        todo!()
    }

    fn as_slice(&self) -> &[Self::Item] {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RustcCanonicalVars;

todo_structural!(RustcCanonicalVars);

impl Default for RustcCanonicalVars {
    fn default() -> Self {
        todo!()
    }
}

pub struct RustcCanonicalVarsIter;
impl Iterator for RustcCanonicalVarsIter {
    type Item = CanonicalVarInfo<RustcInterner>;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl inherent::SliceLike for RustcCanonicalVars {
    type Item = CanonicalVarInfo<RustcInterner>;
    type IntoIter = RustcCanonicalVarsIter;

    fn iter(self) -> Self::IntoIter {
        todo!()
    }

    fn as_slice(&self) -> &[Self::Item] {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RustcExternalConstraints;

todo_structural!(RustcExternalConstraints);

impl std::ops::Deref for RustcExternalConstraints {
    type Target = ExternalConstraintsData<RustcInterner>;

    fn deref(&self) -> &Self::Target {
        todo!()
    }
}

pub struct RustcDepNodeIndex;

#[derive(Debug)]
pub struct RustcTracked<T: fmt::Debug + Clone>(T);

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RustcTys;

todo_structural!(RustcTys);

impl Default for RustcTys {
    fn default() -> Self {
        todo!()
    }
}

pub struct RustcTysIter;
impl Iterator for RustcTysIter {
    type Item = RustcTy;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl inherent::SliceLike for RustcTys {
    type Item = RustcTy;
    type IntoIter = RustcTysIter;

    fn iter(self) -> Self::IntoIter {
        todo!()
    }

    fn as_slice(&self) -> &[Self::Item] {
        todo!()
    }
}

impl inherent::Tys<RustcInterner> for RustcTys {
    fn inputs(self) -> <RustcInterner as rustc_type_ir::Interner>::FnInputTys {
        todo!()
    }

    fn output(self) -> <RustcInterner as rustc_type_ir::Interner>::Ty {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RustcFnInputTys;

todo_structural!(RustcFnInputTys);

impl Default for RustcFnInputTys {
    fn default() -> Self {
        todo!()
    }
}

pub struct RustcFnInputTysIter;
impl Iterator for RustcFnInputTysIter {
    type Item = RustcTy;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl inherent::SliceLike for RustcFnInputTys {
    type Item = RustcTy;
    type IntoIter = RustcFnInputTysIter;

    fn iter(self) -> Self::IntoIter {
        todo!()
    }

    fn as_slice(&self) -> &[Self::Item] {
        todo!()
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct RustcParamTy;

todo_structural!(RustcParamTy);

impl inherent::ParamLike for RustcParamTy {
    fn index(&self) -> u32 {
        todo!()
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct RustcBoundTy;

todo_structural!(RustcBoundTy);

impl inherent::BoundVarLike<RustcInterner> for RustcBoundTy {
    fn var(self) -> rustc_type_ir::BoundVar {
        todo!()
    }

    fn assert_eq(self, var: <RustcInterner as rustc_type_ir::Interner>::BoundVarKind) {
        todo!()
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct RustcPlaceholderTy;

todo_structural!(RustcPlaceholderTy);

impl inherent::PlaceholderLike for RustcPlaceholderTy {
    fn universe(self) -> rustc_type_ir::UniverseIndex {
        todo!()
    }

    fn var(self) -> rustc_type_ir::BoundVar {
        todo!()
    }

    fn with_updated_universe(self, ui: rustc_type_ir::UniverseIndex) -> Self {
        todo!()
    }

    fn new(ui: rustc_type_ir::UniverseIndex, var: rustc_type_ir::BoundVar) -> Self {
        todo!()
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct RustcErrorGuaranteed;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct RustcBoundExistentialPredicates;

todo_structural!(RustcBoundExistentialPredicates);

pub struct RustcBoundExistentialPredicatesIter;
impl Iterator for RustcBoundExistentialPredicatesIter {
    type Item =
        rustc_type_ir::Binder<RustcInterner, rustc_type_ir::ExistentialPredicate<RustcInterner>>;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl inherent::SliceLike for RustcBoundExistentialPredicates {
    type Item =
        rustc_type_ir::Binder<RustcInterner, rustc_type_ir::ExistentialPredicate<RustcInterner>>;
    type IntoIter = RustcBoundExistentialPredicatesIter;

    fn iter(self) -> Self::IntoIter {
        todo!()
    }

    fn as_slice(&self) -> &[Self::Item] {
        todo!()
    }
}

impl inherent::BoundExistentialPredicates<RustcInterner> for RustcBoundExistentialPredicates {
    fn principal_def_id(&self) -> Option<<RustcInterner as rustc_type_ir::Interner>::DefId> {
        todo!()
    }

    fn principal(
        self,
    ) -> Option<
        rustc_type_ir::Binder<RustcInterner, rustc_type_ir::ExistentialTraitRef<RustcInterner>>,
    > {
        todo!()
    }

    fn auto_traits(
        self,
    ) -> impl IntoIterator<Item = <RustcInterner as rustc_type_ir::Interner>::DefId> {
        todo!();
        None
    }

    fn projection_bounds(
        self,
    ) -> impl IntoIterator<
        Item = rustc_type_ir::Binder<
            RustcInterner,
            rustc_type_ir::ExistentialProjection<RustcInterner>,
        >,
    > {
        todo!();
        None
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct RustcAllocId;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct RustcPat;

todo_structural!(RustcPat);

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct RustcSafety;

todo_structural!(RustcSafety);

impl inherent::Safety<RustcInterner> for RustcSafety {
    fn safe() -> Self {
        todo!()
    }

    fn is_safe(self) -> bool {
        todo!()
    }

    fn prefix_str(self) -> &'static str {
        todo!()
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct RustcAbi;

todo_structural!(RustcAbi);

impl inherent::Abi<RustcInterner> for RustcAbi {
    fn rust() -> Self {
        todo!()
    }

    fn is_rust(self) -> bool {
        todo!()
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct RustcPlaceholderConst;

todo_structural!(RustcPlaceholderConst);

impl inherent::PlaceholderLike for RustcPlaceholderConst {
    fn universe(self) -> rustc_type_ir::UniverseIndex {
        todo!()
    }

    fn var(self) -> rustc_type_ir::BoundVar {
        todo!()
    }

    fn with_updated_universe(self, ui: rustc_type_ir::UniverseIndex) -> Self {
        todo!()
    }

    fn new(ui: rustc_type_ir::UniverseIndex, var: rustc_type_ir::BoundVar) -> Self {
        todo!()
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct RustcParamConst;

todo_structural!(RustcParamConst);

impl inherent::ParamLike for RustcParamConst {
    fn index(&self) -> u32 {
        todo!()
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct RustcBoundConst;

todo_structural!(RustcBoundConst);

impl inherent::BoundVarLike<RustcInterner> for RustcBoundConst {
    fn var(self) -> rustc_type_ir::BoundVar {
        todo!()
    }

    fn assert_eq(self, var: <RustcInterner as rustc_type_ir::Interner>::BoundVarKind) {
        todo!()
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct RustcValueConst;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct RustcExprConst;

todo_structural!(RustcExprConst);

impl inherent::ExprConst<RustcInterner> for RustcExprConst {
    fn args(self) -> <RustcInterner as rustc_type_ir::Interner>::GenericArgs {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RustcRegion;

todo_structural!(RustcRegion);

impl inherent::Region<RustcInterner> for RustcRegion {
    fn new_bound(
        interner: RustcInterner,
        debruijn: rustc_type_ir::DebruijnIndex,
        var: <RustcInterner as rustc_type_ir::Interner>::BoundRegion,
    ) -> Self {
        todo!()
    }

    fn new_anon_bound(
        interner: RustcInterner,
        debruijn: rustc_type_ir::DebruijnIndex,
        var: rustc_type_ir::BoundVar,
    ) -> Self {
        todo!()
    }

    fn new_static(interner: RustcInterner) -> Self {
        todo!()
    }
}

impl visit::Flags for RustcRegion {
    fn flags(&self) -> rustc_type_ir::TypeFlags {
        todo!()
    }

    fn outer_exclusive_binder(&self) -> rustc_type_ir::DebruijnIndex {
        todo!()
    }
}

impl fold::TypeSuperFoldable<RustcInterner> for RustcRegion {
    fn try_super_fold_with<F: fold::FallibleTypeFolder<RustcInterner>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        todo!()
    }
}

impl visit::TypeSuperVisitable<RustcInterner> for RustcRegion {
    fn super_visit_with<V: visit::TypeVisitor<RustcInterner>>(&self, visitor: &mut V) -> V::Result {
        todo!()
    }
}

impl inherent::IntoKind for RustcRegion {
    type Kind = rustc_type_ir::RegionKind<RustcInterner>;

    fn kind(self) -> Self::Kind {
        todo!()
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct RustcEarlyParamRegion;

impl inherent::ParamLike for RustcEarlyParamRegion {
    fn index(&self) -> u32 {
        todo!()
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct RustcLateParamRegion;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct RustcBoundRegion;

todo_structural!(RustcBoundRegion);

impl inherent::BoundVarLike<RustcInterner> for RustcBoundRegion {
    fn var(self) -> rustc_type_ir::BoundVar {
        todo!()
    }

    fn assert_eq(self, var: <RustcInterner as rustc_type_ir::Interner>::BoundVarKind) {
        todo!()
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct RustcPlaceholderRegion;

todo_structural!(RustcPlaceholderRegion);

impl inherent::PlaceholderLike for RustcPlaceholderRegion {
    fn universe(self) -> rustc_type_ir::UniverseIndex {
        todo!()
    }

    fn var(self) -> rustc_type_ir::BoundVar {
        todo!()
    }

    fn with_updated_universe(self, ui: rustc_type_ir::UniverseIndex) -> Self {
        todo!()
    }

    fn new(ui: rustc_type_ir::UniverseIndex, var: rustc_type_ir::BoundVar) -> Self {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RustcParamEnv;

todo_structural!(RustcParamEnv);

impl inherent::ParamEnv<RustcInterner> for RustcParamEnv {
    fn reveal(&self) -> rustc_type_ir::solve::Reveal {
        todo!()
    }

    fn caller_bounds(
        self,
    ) -> impl IntoIterator<Item = <RustcInterner as rustc_type_ir::Interner>::Clause> {
        todo!();
        None
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RustcPredicate;

todo_structural!(RustcPredicate);

impl inherent::Predicate<RustcInterner> for RustcPredicate {
    fn as_clause(self) -> Option<<RustcInterner as rustc_type_ir::Interner>::Clause> {
        todo!()
    }

    fn is_coinductive(&self, interner: RustcInterner) -> bool {
        todo!()
    }

    fn allow_normalization(&self) -> bool {
        todo!()
    }
}

impl elaborate::Elaboratable<RustcInterner> for RustcPredicate {
    fn predicate_kind(
        self,
    ) -> rustc_type_ir::Binder<RustcInterner, rustc_type_ir::PredicateKind<RustcInterner>> {
        todo!()
    }

    fn as_clause(self) -> Option<<RustcInterner as rustc_type_ir::Interner>::Clause> {
        todo!()
    }

    fn child(&self, clause: <RustcInterner as rustc_type_ir::Interner>::Clause) -> Self {
        todo!()
    }

    fn child_with_derived_cause(
        &self,
        clause: <RustcInterner as rustc_type_ir::Interner>::Clause,
        span: <RustcInterner as rustc_type_ir::Interner>::Span,
        parent_trait_pred: rustc_type_ir::Binder<
            RustcInterner,
            rustc_type_ir::TraitPredicate<RustcInterner>,
        >,
        index: usize,
    ) -> Self {
        todo!()
    }
}

impl inherent::IntoKind for RustcPredicate {
    type Kind = rustc_type_ir::Binder<RustcInterner, rustc_type_ir::PredicateKind<RustcInterner>>;

    fn kind(self) -> Self::Kind {
        todo!()
    }
}

impl visit::Flags for RustcPredicate {
    fn flags(&self) -> rustc_type_ir::TypeFlags {
        todo!()
    }

    fn outer_exclusive_binder(&self) -> rustc_type_ir::DebruijnIndex {
        todo!()
    }
}

impl fold::TypeSuperFoldable<RustcInterner> for RustcPredicate {
    fn try_super_fold_with<F: fold::FallibleTypeFolder<RustcInterner>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        todo!()
    }
}

impl visit::TypeSuperVisitable<RustcInterner> for RustcPredicate {
    fn super_visit_with<V: visit::TypeVisitor<RustcInterner>>(&self, visitor: &mut V) -> V::Result {
        todo!()
    }
}

impl rustc_type_ir::UpcastFrom<RustcInterner, rustc_type_ir::PredicateKind<RustcInterner>>
    for RustcPredicate
{
    fn upcast_from(
        from: rustc_type_ir::PredicateKind<RustcInterner>,
        interner: RustcInterner,
    ) -> Self {
        todo!()
    }
}

impl
    rustc_type_ir::UpcastFrom<
        RustcInterner,
        rustc_type_ir::Binder<RustcInterner, rustc_type_ir::PredicateKind<RustcInterner>>,
    > for RustcPredicate
{
    fn upcast_from(
        from: rustc_type_ir::Binder<RustcInterner, rustc_type_ir::PredicateKind<RustcInterner>>,
        interner: RustcInterner,
    ) -> Self {
        todo!()
    }
}

impl rustc_type_ir::UpcastFrom<RustcInterner, rustc_type_ir::ClauseKind<RustcInterner>>
    for RustcPredicate
{
    fn upcast_from(
        from: rustc_type_ir::ClauseKind<RustcInterner>,
        interner: RustcInterner,
    ) -> Self {
        todo!()
    }
}

impl
    rustc_type_ir::UpcastFrom<
        RustcInterner,
        rustc_type_ir::Binder<RustcInterner, rustc_type_ir::ClauseKind<RustcInterner>>,
    > for RustcPredicate
{
    fn upcast_from(
        from: rustc_type_ir::Binder<RustcInterner, rustc_type_ir::ClauseKind<RustcInterner>>,
        interner: RustcInterner,
    ) -> Self {
        todo!()
    }
}

impl rustc_type_ir::UpcastFrom<RustcInterner, RustcClause> for RustcPredicate {
    fn upcast_from(from: RustcClause, interner: RustcInterner) -> Self {
        todo!()
    }
}

impl rustc_type_ir::UpcastFrom<RustcInterner, rustc_type_ir::NormalizesTo<RustcInterner>>
    for RustcPredicate
{
    fn upcast_from(
        from: rustc_type_ir::NormalizesTo<RustcInterner>,
        interner: RustcInterner,
    ) -> Self {
        todo!()
    }
}

impl rustc_type_ir::UpcastFrom<RustcInterner, rustc_type_ir::TraitRef<RustcInterner>>
    for RustcPredicate
{
    fn upcast_from(from: rustc_type_ir::TraitRef<RustcInterner>, interner: RustcInterner) -> Self {
        todo!()
    }
}

impl
    rustc_type_ir::UpcastFrom<
        RustcInterner,
        rustc_type_ir::Binder<RustcInterner, rustc_type_ir::TraitRef<RustcInterner>>,
    > for RustcPredicate
{
    fn upcast_from(
        from: rustc_type_ir::Binder<RustcInterner, rustc_type_ir::TraitRef<RustcInterner>>,
        interner: RustcInterner,
    ) -> Self {
        todo!()
    }
}

impl rustc_type_ir::UpcastFrom<RustcInterner, rustc_type_ir::TraitPredicate<RustcInterner>>
    for RustcPredicate
{
    fn upcast_from(
        from: rustc_type_ir::TraitPredicate<RustcInterner>,
        interner: RustcInterner,
    ) -> Self {
        todo!()
    }
}

impl
    rustc_type_ir::UpcastFrom<
        RustcInterner,
        rustc_type_ir::OutlivesPredicate<RustcInterner, RustcTy>,
    > for RustcPredicate
{
    fn upcast_from(
        from: rustc_type_ir::OutlivesPredicate<RustcInterner, RustcTy>,
        interner: RustcInterner,
    ) -> Self {
        todo!()
    }
}

impl
    rustc_type_ir::UpcastFrom<
        RustcInterner,
        rustc_type_ir::OutlivesPredicate<RustcInterner, RustcRegion>,
    > for RustcPredicate
{
    fn upcast_from(
        from: rustc_type_ir::OutlivesPredicate<RustcInterner, RustcRegion>,
        interner: RustcInterner,
    ) -> Self {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RustcClause;

todo_structural!(RustcClause);

impl inherent::Clause<RustcInterner> for RustcClause {
    fn as_predicate(self) -> <RustcInterner as rustc_type_ir::Interner>::Predicate {
        todo!()
    }

    fn instantiate_supertrait(
        self,
        cx: RustcInterner,
        trait_ref: rustc_type_ir::Binder<RustcInterner, rustc_type_ir::TraitRef<RustcInterner>>,
    ) -> Self {
        todo!()
    }
}

impl elaborate::Elaboratable<RustcInterner> for RustcClause {
    fn predicate_kind(
        self,
    ) -> rustc_type_ir::Binder<RustcInterner, rustc_type_ir::PredicateKind<RustcInterner>> {
        todo!()
    }

    fn as_clause(self) -> Option<<RustcInterner as rustc_type_ir::Interner>::Clause> {
        todo!()
    }

    fn child(&self, clause: <RustcInterner as rustc_type_ir::Interner>::Clause) -> Self {
        todo!()
    }

    fn child_with_derived_cause(
        &self,
        clause: <RustcInterner as rustc_type_ir::Interner>::Clause,
        span: <RustcInterner as rustc_type_ir::Interner>::Span,
        parent_trait_pred: rustc_type_ir::Binder<
            RustcInterner,
            rustc_type_ir::TraitPredicate<RustcInterner>,
        >,
        index: usize,
    ) -> Self {
        todo!()
    }
}

impl inherent::IntoKind for RustcClause {
    type Kind = rustc_type_ir::Binder<RustcInterner, rustc_type_ir::ClauseKind<RustcInterner>>;

    fn kind(self) -> Self::Kind {
        todo!()
    }
}

impl visit::Flags for RustcClause {
    fn flags(&self) -> rustc_type_ir::TypeFlags {
        todo!()
    }

    fn outer_exclusive_binder(&self) -> rustc_type_ir::DebruijnIndex {
        todo!()
    }
}

impl fold::TypeSuperFoldable<RustcInterner> for RustcClause {
    fn try_super_fold_with<F: fold::FallibleTypeFolder<RustcInterner>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        todo!()
    }
}

impl visit::TypeSuperVisitable<RustcInterner> for RustcClause {
    fn super_visit_with<V: visit::TypeVisitor<RustcInterner>>(&self, visitor: &mut V) -> V::Result {
        todo!()
    }
}

impl
    rustc_type_ir::UpcastFrom<
        RustcInterner,
        rustc_type_ir::Binder<RustcInterner, rustc_type_ir::ClauseKind<RustcInterner>>,
    > for RustcClause
{
    fn upcast_from(
        from: rustc_type_ir::Binder<RustcInterner, rustc_type_ir::ClauseKind<RustcInterner>>,
        interner: RustcInterner,
    ) -> Self {
        todo!()
    }
}

impl rustc_type_ir::UpcastFrom<RustcInterner, rustc_type_ir::TraitRef<RustcInterner>>
    for RustcClause
{
    fn upcast_from(from: rustc_type_ir::TraitRef<RustcInterner>, interner: RustcInterner) -> Self {
        todo!()
    }
}

impl
    rustc_type_ir::UpcastFrom<
        RustcInterner,
        rustc_type_ir::Binder<RustcInterner, rustc_type_ir::TraitRef<RustcInterner>>,
    > for RustcClause
{
    fn upcast_from(
        from: rustc_type_ir::Binder<RustcInterner, rustc_type_ir::TraitRef<RustcInterner>>,
        interner: RustcInterner,
    ) -> Self {
        todo!()
    }
}

impl rustc_type_ir::UpcastFrom<RustcInterner, rustc_type_ir::TraitPredicate<RustcInterner>>
    for RustcClause
{
    fn upcast_from(
        from: rustc_type_ir::TraitPredicate<RustcInterner>,
        interner: RustcInterner,
    ) -> Self {
        todo!()
    }
}

impl
    rustc_type_ir::UpcastFrom<
        RustcInterner,
        rustc_type_ir::Binder<RustcInterner, rustc_type_ir::TraitPredicate<RustcInterner>>,
    > for RustcClause
{
    fn upcast_from(
        from: rustc_type_ir::Binder<RustcInterner, rustc_type_ir::TraitPredicate<RustcInterner>>,
        interner: RustcInterner,
    ) -> Self {
        todo!()
    }
}

impl rustc_type_ir::UpcastFrom<RustcInterner, rustc_type_ir::ProjectionPredicate<RustcInterner>>
    for RustcClause
{
    fn upcast_from(
        from: rustc_type_ir::ProjectionPredicate<RustcInterner>,
        interner: RustcInterner,
    ) -> Self {
        todo!()
    }
}

impl
    rustc_type_ir::UpcastFrom<
        RustcInterner,
        rustc_type_ir::Binder<RustcInterner, rustc_type_ir::ProjectionPredicate<RustcInterner>>,
    > for RustcClause
{
    fn upcast_from(
        from: rustc_type_ir::Binder<
            RustcInterner,
            rustc_type_ir::ProjectionPredicate<RustcInterner>,
        >,
        interner: RustcInterner,
    ) -> Self {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RustcClauses;

todo_structural!(RustcClauses);

pub struct RustcClausesIter;
impl Iterator for RustcClausesIter {
    type Item = ();

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl inherent::SliceLike for RustcClauses {
    type Item = ();
    type IntoIter = RustcClausesIter;

    fn iter(self) -> Self::IntoIter {
        todo!()
    }

    fn as_slice(&self) -> &[Self::Item] {
        todo!()
    }
}

impl visit::Flags for RustcClauses {
    fn flags(&self) -> rustc_type_ir::TypeFlags {
        todo!()
    }

    fn outer_exclusive_binder(&self) -> rustc_type_ir::DebruijnIndex {
        todo!()
    }
}

impl fold::TypeSuperFoldable<RustcInterner> for RustcClauses {
    fn try_super_fold_with<F: fold::FallibleTypeFolder<RustcInterner>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        todo!()
    }
}

impl visit::TypeSuperVisitable<RustcInterner> for RustcClauses {
    fn super_visit_with<V: visit::TypeVisitor<RustcInterner>>(&self, visitor: &mut V) -> V::Result {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RustcGenericsOf;

todo_structural!(RustcGenericsOf);

impl inherent::GenericsOf<RustcInterner> for RustcGenericsOf {
    fn count(&self) -> usize {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RustcVariancesOf;

todo_structural!(RustcVariancesOf);

pub struct RustcVariancesOfIter;
impl Iterator for RustcVariancesOfIter {
    type Item = rustc_type_ir::Variance;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl inherent::SliceLike for RustcVariancesOf {
    type Item = rustc_type_ir::Variance;
    type IntoIter = RustcVariancesOfIter;

    fn iter(self) -> Self::IntoIter {
        todo!()
    }

    fn as_slice(&self) -> &[Self::Item] {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RustcAdtDef;

todo_structural!(RustcAdtDef);

impl inherent::AdtDef<RustcInterner> for RustcAdtDef {
    fn def_id(&self) -> <RustcInterner as rustc_type_ir::Interner>::DefId {
        todo!()
    }

    fn is_struct(&self) -> bool {
        todo!()
    }

    fn struct_tail_ty(
        self,
        interner: RustcInterner,
    ) -> Option<
        rustc_type_ir::EarlyBinder<RustcInterner, <RustcInterner as rustc_type_ir::Interner>::Ty>,
    > {
        todo!()
    }

    fn is_phantom_data(&self) -> bool {
        todo!()
    }

    fn all_field_tys(
        self,
        interner: RustcInterner,
    ) -> rustc_type_ir::EarlyBinder<
        RustcInterner,
        impl IntoIterator<Item = <RustcInterner as rustc_type_ir::Interner>::Ty>,
    > {
        todo!();
        rustc_type_ir::EarlyBinder::bind(None)
    }

    fn sized_constraint(
        self,
        interner: RustcInterner,
    ) -> Option<
        rustc_type_ir::EarlyBinder<RustcInterner, <RustcInterner as rustc_type_ir::Interner>::Ty>,
    > {
        todo!()
    }

    fn is_fundamental(&self) -> bool {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RustcFeatures;

impl inherent::Features<RustcInterner> for RustcFeatures {
    fn generic_const_exprs(self) -> bool {
        todo!()
    }

    fn coroutine_clone(self) -> bool {
        todo!()
    }

    fn associated_const_equality(self) -> bool {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RustcUnsizingParams;

impl std::ops::Deref for RustcUnsizingParams {
    type Target = BitSet<u32>;

    fn deref(&self) -> &Self::Target {
        todo!()
    }
}

impl rustc_type_ir::Interner for RustcInterner {
    type DefId = InternId;
    type LocalDefId = InternId;
    type Span = RustcSpan;

    type GenericArgs = GenericArgs;
    type GenericArgsSlice = GenericArgs;
    type GenericArg = RustcGenericArg;

    type Term = RustcTerm;

    type BoundVarKinds = RustcBoundVarKinds;
    type BoundVarKind = RustcBoundVarKind;

    type PredefinedOpaques = RustcPredefinedOpaques;

    fn mk_predefined_opaques_in_body(
        self,
        data: rustc_type_ir::solve::PredefinedOpaquesData<Self>,
    ) -> Self::PredefinedOpaques {
        todo!()
    }

    type DefiningOpaqueTypes = RustcDefiningOpaqueTypes;

    type CanonicalVars = RustcCanonicalVars;

    fn mk_canonical_var_infos(
        self,
        infos: &[rustc_type_ir::CanonicalVarInfo<Self>],
    ) -> Self::CanonicalVars {
        todo!()
    }

    type ExternalConstraints = RustcExternalConstraints;

    fn mk_external_constraints(
        self,
        data: rustc_type_ir::solve::ExternalConstraintsData<Self>,
    ) -> Self::ExternalConstraints {
        todo!()
    }

    type DepNodeIndex = RustcDepNodeIndex;

    type Tracked<T: fmt::Debug + Clone> = RustcTracked<T>;

    fn mk_tracked<T: fmt::Debug + Clone>(
        self,
        data: T,
        dep_node: Self::DepNodeIndex,
    ) -> Self::Tracked<T> {
        todo!()
    }

    fn get_tracked<T: fmt::Debug + Clone>(self, tracked: &Self::Tracked<T>) -> T {
        todo!()
    }

    fn with_cached_task<T>(self, task: impl FnOnce() -> T) -> (T, Self::DepNodeIndex) {
        todo!()
    }

    type Ty = RustcTy;
    type Tys = RustcTys;
    type FnInputTys = RustcFnInputTys;
    type ParamTy = RustcParamTy;
    type BoundTy = RustcBoundTy;
    type PlaceholderTy = RustcPlaceholderTy;

    type ErrorGuaranteed = RustcErrorGuaranteed;
    type BoundExistentialPredicates = RustcBoundExistentialPredicates;
    type AllocId = RustcAllocId;
    type Pat = RustcPat;
    type Safety = RustcSafety;
    type Abi = RustcAbi;

    type Const = RustcConst;
    type PlaceholderConst = RustcPlaceholderConst;
    type ParamConst = RustcParamConst;
    type BoundConst = RustcBoundConst;
    type ValueConst = RustcValueConst;
    type ExprConst = RustcExprConst;

    type Region = RustcRegion;
    type EarlyParamRegion = RustcEarlyParamRegion;
    type LateParamRegion = RustcLateParamRegion;
    type BoundRegion = RustcBoundRegion;
    type PlaceholderRegion = RustcPlaceholderRegion;

    type ParamEnv = RustcParamEnv;
    type Predicate = RustcPredicate;
    type Clause = RustcClause;
    type Clauses = RustcClauses;

    fn with_global_cache<R>(
        self,
        f: impl FnOnce(&mut rustc_type_ir::search_graph::GlobalCache<Self>) -> R,
    ) -> R {
        todo!()
    }

    fn evaluation_is_concurrent(&self) -> bool {
        todo!()
    }

    fn expand_abstract_consts<T: rustc_type_ir::fold::TypeFoldable<Self>>(self, t: T) -> T {
        todo!()
    }

    type GenericsOf = RustcGenericsOf;

    fn generics_of(self, def_id: Self::DefId) -> Self::GenericsOf {
        todo!()
    }

    type VariancesOf = RustcVariancesOf;

    fn variances_of(self, def_id: Self::DefId) -> Self::VariancesOf {
        todo!()
    }

    fn type_of(self, def_id: Self::DefId) -> rustc_type_ir::EarlyBinder<Self, Self::Ty> {
        todo!()
    }

    type AdtDef = RustcAdtDef;

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
        todo!()
    }

    fn mk_args_from_iter<I, T>(self, args: I) -> T::Output
    where
        I: Iterator<Item = T>,
        T: rustc_type_ir::CollectAndApply<Self::GenericArg, Self::GenericArgs>,
    {
        todo!()
    }

    fn check_args_compatible(self, def_id: Self::DefId, args: Self::GenericArgs) -> bool {
        todo!()
    }

    fn debug_assert_args_compatible(self, def_id: Self::DefId, args: Self::GenericArgs) {
        todo!()
    }

    fn debug_assert_existential_args_compatible(
        self,
        def_id: Self::DefId,
        args: Self::GenericArgs,
    ) {
        todo!()
    }

    fn mk_type_list_from_iter<I, T>(self, args: I) -> T::Output
    where
        I: Iterator<Item = T>,
        T: rustc_type_ir::CollectAndApply<Self::Ty, Self::Tys>,
    {
        todo!()
    }

    fn parent(self, def_id: Self::DefId) -> Self::DefId {
        todo!()
    }

    fn recursion_limit(self) -> usize {
        todo!()
    }

    type Features = RustcFeatures;

    fn features(self) -> Self::Features {
        todo!()
    }

    fn bound_coroutine_hidden_types(
        self,
        def_id: Self::DefId,
    ) -> impl IntoIterator<Item = rustc_type_ir::EarlyBinder<Self, rustc_type_ir::Binder<Self, Self::Ty>>>
    {
        todo!();
        None
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
        todo!();
        rustc_type_ir::EarlyBinder::bind(None)
    }

    fn predicates_of(
        self,
        def_id: Self::DefId,
    ) -> rustc_type_ir::EarlyBinder<Self, impl IntoIterator<Item = Self::Clause>> {
        todo!();
        rustc_type_ir::EarlyBinder::bind(None)
    }

    fn own_predicates_of(
        self,
        def_id: Self::DefId,
    ) -> rustc_type_ir::EarlyBinder<Self, impl IntoIterator<Item = Self::Clause>> {
        todo!();
        rustc_type_ir::EarlyBinder::bind(None)
    }

    fn explicit_super_predicates_of(
        self,
        def_id: Self::DefId,
    ) -> rustc_type_ir::EarlyBinder<Self, impl IntoIterator<Item = (Self::Clause, Self::Span)>>
    {
        todo!();
        rustc_type_ir::EarlyBinder::bind(None)
    }

    fn explicit_implied_predicates_of(
        self,
        def_id: Self::DefId,
    ) -> rustc_type_ir::EarlyBinder<Self, impl IntoIterator<Item = (Self::Clause, Self::Span)>>
    {
        todo!();
        rustc_type_ir::EarlyBinder::bind(None)
    }

    fn is_const_impl(self, def_id: Self::DefId) -> bool {
        todo!()
    }

    fn const_conditions(
        self,
        def_id: Self::DefId,
    ) -> rustc_type_ir::EarlyBinder<
        Self,
        impl IntoIterator<Item = rustc_type_ir::Binder<Self, rustc_type_ir::TraitRef<Self>>>,
    > {
        todo!();
        rustc_type_ir::EarlyBinder::bind(None)
    }

    fn implied_const_bounds(
        self,
        def_id: Self::DefId,
    ) -> rustc_type_ir::EarlyBinder<
        Self,
        impl IntoIterator<Item = rustc_type_ir::Binder<Self, rustc_type_ir::TraitRef<Self>>>,
    > {
        todo!();
        rustc_type_ir::EarlyBinder::bind(None)
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
        todo!();
        None
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

    fn layout_is_pointer_like(self, param_env: Self::ParamEnv, ty: Self::Ty) -> bool {
        todo!()
    }

    type UnsizingParams = RustcUnsizingParams;

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
}
