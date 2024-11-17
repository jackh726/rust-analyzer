#![allow(unused)]

use base_db::CrateId;
use chalk_ir::{ProgramClauseImplication, SeparatorTraitRef, Variance};
use hir_def::{AdtId, BlockId, GenericDefId, TypeAliasId, VariantId};
use intern::{impl_internable, Interned};
use smallvec::{smallvec, SmallVec};
use std::fmt;
use triomphe::Arc;

use rustc_ast_ir::visit::VisitorResult;
use rustc_index_in_tree::{bit_set::BitSet, IndexVec};
use rustc_type_ir::{
    elaborate, fold, inherent, ir_print, relate,
    solve::{ExternalConstraintsData, PredefinedOpaquesData},
    visit, CanonicalVarInfo, ConstKind, GenericArgKind, RegionKind, RustIr, TermKind, TyKind,
};

use crate::{
    db::HirDatabase, generics::generics, interner::InternedWrapper, ConstScalar, FnAbi, Interner,
};

use super::mapping::{convert_binder_to_early_binder, ChalkToNextSolver};

impl_internable!(
    InternedWrapper<rustc_type_ir::ConstKind<DbInterner>>,
    InternedWrapper<rustc_type_ir::RegionKind<DbInterner>>,
    InternedWrapper<rustc_type_ir::TyKind<DbInterner>>,
    InternedWrapper<SmallVec<[GenericArg; 2]>>,
    InternedWrapper<SmallVec<[Ty; 2]>>,
);

#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct DbInterner;

macro_rules! todo_structural {
    ($t:ty) => {
        impl relate::Relate<DbInterner> for $t {
            fn relate<R: relate::TypeRelation>(
                _relation: &mut R,
                _a: Self,
                _b: Self,
            ) -> relate::RelateResult<DbInterner, Self> {
                todo!()
            }
        }

        impl fold::TypeFoldable<DbInterner> for $t {
            fn try_fold_with<F: fold::FallibleTypeFolder<DbInterner>>(
                self,
                _folder: &mut F,
            ) -> Result<Self, F::Error> {
                todo!()
            }
        }

        impl visit::TypeVisitable<DbInterner> for $t {
            fn visit_with<V: visit::TypeVisitor<DbInterner>>(&self, _visitor: &mut V) -> V::Result {
                todo!()
            }
        }
    };
}

impl inherent::DefId<DbInterner> for GenericDefId {
    fn as_local(self) -> Option<GenericDefId> {
        Some(self)
    }
    fn is_local(self) -> bool {
        true
    }
}

todo_structural!(GenericDefId);

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct Span(Option<span::Span>);

todo_structural!(Span);

impl inherent::Span<DbInterner> for Span {
    fn dummy() -> Self {
        Span(None)
    }
}

type InternedGenericArgs = Interned<InternedWrapper<SmallVec<[GenericArg; 2]>>>;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct GenericArgs(InternedGenericArgs);

todo_structural!(GenericArgs);

impl GenericArgs {
    pub fn new(data: impl IntoIterator<Item = GenericArg>) -> Self {
        GenericArgs(Interned::new(InternedWrapper(data.into_iter().collect())))
    }
}

impl inherent::GenericArgs<DbInterner> for GenericArgs {
    fn dummy() -> Self {
        GenericArgs(Interned::new(InternedWrapper(smallvec![])))
    }

    fn rebase_onto(
        self,
        interner: DbInterner,
        source_def_id: <DbInterner as rustc_type_ir::Interner>::DefId,
        target: <DbInterner as rustc_type_ir::Interner>::GenericArgs,
    ) -> <DbInterner as rustc_type_ir::Interner>::GenericArgs {
        todo!()
    }

    fn type_at(self, i: usize) -> <DbInterner as rustc_type_ir::Interner>::Ty {
        todo!()
    }

    fn region_at(self, i: usize) -> <DbInterner as rustc_type_ir::Interner>::Region {
        todo!()
    }

    fn const_at(self, i: usize) -> <DbInterner as rustc_type_ir::Interner>::Const {
        todo!()
    }

    fn identity_for_item(
        interner: DbInterner,
        def_id: <DbInterner as rustc_type_ir::Interner>::DefId,
    ) -> <DbInterner as rustc_type_ir::Interner>::GenericArgs {
        todo!()
    }

    fn extend_with_error(
        interner: DbInterner,
        def_id: <DbInterner as rustc_type_ir::Interner>::DefId,
        original_args: &[GenericArg],
    ) -> <DbInterner as rustc_type_ir::Interner>::GenericArgs {
        todo!()
    }

    fn split_closure_args(self) -> rustc_type_ir::ClosureArgsParts<DbInterner> {
        todo!()
    }

    fn split_coroutine_closure_args(self) -> rustc_type_ir::CoroutineClosureArgsParts<DbInterner> {
        todo!()
    }

    fn split_coroutine_args(self) -> rustc_type_ir::CoroutineArgsParts<DbInterner> {
        todo!()
    }
}

pub struct GenericArgsIter;
impl Iterator for GenericArgsIter {
    type Item = GenericArg;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl inherent::SliceLike for GenericArgs {
    type Item = GenericArg;
    type IntoIter = GenericArgsIter;

    fn iter(self) -> Self::IntoIter {
        todo!()
    }

    fn as_slice(&self) -> &[Self::Item] {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct GenericArg;

todo_structural!(GenericArg);

impl inherent::GenericArg<DbInterner> for GenericArg {}

impl inherent::IntoKind for GenericArg {
    type Kind = GenericArgKind<DbInterner>;

    fn kind(self) -> Self::Kind {
        todo!()
    }
}

impl From<Ty> for GenericArg {
    fn from(value: Ty) -> Self {
        todo!()
    }
}

impl From<Const> for GenericArg {
    fn from(value: Const) -> Self {
        todo!()
    }
}

impl From<Region> for GenericArg {
    fn from(value: Region) -> Self {
        todo!()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Ty(Interned<InternedWrapper<rustc_type_ir::TyKind<DbInterner>>>);

impl Ty {
    pub fn new(kind: rustc_type_ir::TyKind<DbInterner>) -> Self {
        Ty(Interned::new(InternedWrapper(kind)))
    }

    pub fn from_chalk(kind: chalk_ir::TyKind<Interner>) -> Self {
        todo!()
    }
}

impl PartialOrd for Ty {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        todo!()
    }
}

impl Ord for Ty {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        todo!()
    }
}

todo_structural!(Ty);

impl inherent::Ty<DbInterner> for Ty {
    fn new_unit(interner: DbInterner) -> Self {
        todo!()
    }

    fn new_bool(interner: DbInterner) -> Self {
        todo!()
    }

    fn new_u8(interner: DbInterner) -> Self {
        todo!()
    }

    fn new_usize(interner: DbInterner) -> Self {
        todo!()
    }

    fn new_infer(interner: DbInterner, var: rustc_type_ir::InferTy) -> Self {
        todo!()
    }

    fn new_var(interner: DbInterner, var: rustc_type_ir::TyVid) -> Self {
        todo!()
    }

    fn new_param(
        interner: DbInterner,
        param: <DbInterner as rustc_type_ir::Interner>::ParamTy,
    ) -> Self {
        let kind = rustc_type_ir::TyKind::Param(param);
        Ty(Interned::new(InternedWrapper(kind)))
    }

    fn new_placeholder(
        interner: DbInterner,
        param: <DbInterner as rustc_type_ir::Interner>::PlaceholderTy,
    ) -> Self {
        todo!()
    }

    fn new_bound(
        interner: DbInterner,
        debruijn: rustc_type_ir::DebruijnIndex,
        var: <DbInterner as rustc_type_ir::Interner>::BoundTy,
    ) -> Self {
        todo!()
    }

    fn new_anon_bound(
        interner: DbInterner,
        debruijn: rustc_type_ir::DebruijnIndex,
        var: rustc_type_ir::BoundVar,
    ) -> Self {
        todo!()
    }

    fn new_alias(
        interner: DbInterner,
        kind: rustc_type_ir::AliasTyKind,
        alias_ty: rustc_type_ir::AliasTy<DbInterner>,
    ) -> Self {
        todo!()
    }

    fn new_error(
        interner: DbInterner,
        guar: <DbInterner as rustc_type_ir::Interner>::ErrorGuaranteed,
    ) -> Self {
        todo!()
    }

    fn new_adt(
        interner: DbInterner,
        adt_def: <DbInterner as rustc_type_ir::Interner>::AdtDef,
        args: <DbInterner as rustc_type_ir::Interner>::GenericArgs,
    ) -> Self {
        todo!()
    }

    fn new_foreign(
        interner: DbInterner,
        def_id: <DbInterner as rustc_type_ir::Interner>::DefId,
    ) -> Self {
        todo!()
    }

    fn new_dynamic(
        interner: DbInterner,
        preds: <DbInterner as rustc_type_ir::Interner>::BoundExistentialPredicates,
        region: <DbInterner as rustc_type_ir::Interner>::Region,
        kind: rustc_type_ir::DynKind,
    ) -> Self {
        todo!()
    }

    fn new_coroutine(
        interner: DbInterner,
        def_id: <DbInterner as rustc_type_ir::Interner>::DefId,
        args: <DbInterner as rustc_type_ir::Interner>::GenericArgs,
    ) -> Self {
        todo!()
    }

    fn new_coroutine_closure(
        interner: DbInterner,
        def_id: <DbInterner as rustc_type_ir::Interner>::DefId,
        args: <DbInterner as rustc_type_ir::Interner>::GenericArgs,
    ) -> Self {
        todo!()
    }

    fn new_closure(
        interner: DbInterner,
        def_id: <DbInterner as rustc_type_ir::Interner>::DefId,
        args: <DbInterner as rustc_type_ir::Interner>::GenericArgs,
    ) -> Self {
        todo!()
    }

    fn new_coroutine_witness(
        interner: DbInterner,
        def_id: <DbInterner as rustc_type_ir::Interner>::DefId,
        args: <DbInterner as rustc_type_ir::Interner>::GenericArgs,
    ) -> Self {
        todo!()
    }

    fn new_ptr(interner: DbInterner, ty: Self, mutbl: rustc_ast_ir::Mutability) -> Self {
        todo!()
    }

    fn new_ref(
        interner: DbInterner,
        region: <DbInterner as rustc_type_ir::Interner>::Region,
        ty: Self,
        mutbl: rustc_ast_ir::Mutability,
    ) -> Self {
        todo!()
    }

    fn new_array_with_const_len(
        interner: DbInterner,
        ty: Self,
        len: <DbInterner as rustc_type_ir::Interner>::Const,
    ) -> Self {
        todo!()
    }

    fn new_slice(interner: DbInterner, ty: Self) -> Self {
        todo!()
    }

    fn new_tup(interner: DbInterner, tys: &[<DbInterner as rustc_type_ir::Interner>::Ty]) -> Self {
        todo!()
    }

    fn new_tup_from_iter<It, T>(interner: DbInterner, iter: It) -> T::Output
    where
        It: Iterator<Item = T>,
        T: rustc_type_ir::CollectAndApply<Self, Self>,
    {
        todo!()
    }

    fn new_fn_def(
        interner: DbInterner,
        def_id: <DbInterner as rustc_type_ir::Interner>::DefId,
        args: <DbInterner as rustc_type_ir::Interner>::GenericArgs,
    ) -> Self {
        todo!()
    }

    fn new_fn_ptr(
        interner: DbInterner,
        sig: rustc_type_ir::Binder<DbInterner, rustc_type_ir::FnSig<DbInterner>>,
    ) -> Self {
        todo!()
    }

    fn new_pat(
        interner: DbInterner,
        ty: Self,
        pat: <DbInterner as rustc_type_ir::Interner>::Pat,
    ) -> Self {
        todo!()
    }

    fn tuple_fields(self) -> <DbInterner as rustc_type_ir::Interner>::Tys {
        todo!()
    }

    fn to_opt_closure_kind(self) -> Option<rustc_type_ir::ClosureKind> {
        todo!()
    }

    fn from_closure_kind(interner: DbInterner, kind: rustc_type_ir::ClosureKind) -> Self {
        todo!()
    }

    fn from_coroutine_closure_kind(interner: DbInterner, kind: rustc_type_ir::ClosureKind) -> Self {
        todo!()
    }

    fn discriminant_ty(self, interner: DbInterner) -> <DbInterner as rustc_type_ir::Interner>::Ty {
        todo!()
    }

    fn async_destructor_ty(
        self,
        interner: DbInterner,
    ) -> <DbInterner as rustc_type_ir::Interner>::Ty {
        todo!()
    }
}

impl visit::Flags for Ty {
    fn flags(&self) -> rustc_type_ir::TypeFlags {
        todo!()
    }

    fn outer_exclusive_binder(&self) -> rustc_type_ir::DebruijnIndex {
        todo!()
    }
}

impl fold::TypeSuperFoldable<DbInterner> for Ty {
    fn try_super_fold_with<F: fold::FallibleTypeFolder<DbInterner>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        todo!()
    }
}

impl visit::TypeSuperVisitable<DbInterner> for Ty {
    fn super_visit_with<V: visit::TypeVisitor<DbInterner>>(&self, visitor: &mut V) -> V::Result {
        todo!()
    }
}

impl inherent::IntoKind for Ty {
    type Kind = rustc_type_ir::TyKind<DbInterner>;

    fn kind(self) -> Self::Kind {
        todo!()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Const(Interned<InternedWrapper<rustc_type_ir::ConstKind<DbInterner>>>);

impl Const {
    pub fn new(kind: rustc_type_ir::ConstKind<DbInterner>) -> Self {
        Const(Interned::new(InternedWrapper(kind)))
    }
}

impl PartialOrd for Const {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        todo!()
    }
}

impl Ord for Const {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        todo!()
    }
}

todo_structural!(Const);

impl inherent::Const<DbInterner> for Const {
    fn try_to_target_usize(self, interner: DbInterner) -> Option<u64> {
        todo!()
    }

    fn new_infer(interner: DbInterner, var: rustc_type_ir::InferConst) -> Self {
        todo!()
    }

    fn new_var(interner: DbInterner, var: rustc_type_ir::ConstVid) -> Self {
        todo!()
    }

    fn new_bound(
        interner: DbInterner,
        debruijn: rustc_type_ir::DebruijnIndex,
        var: <DbInterner as rustc_type_ir::Interner>::BoundConst,
    ) -> Self {
        todo!()
    }

    fn new_anon_bound(
        interner: DbInterner,
        debruijn: rustc_type_ir::DebruijnIndex,
        var: rustc_type_ir::BoundVar,
    ) -> Self {
        todo!()
    }

    fn new_unevaluated(
        interner: DbInterner,
        uv: rustc_type_ir::UnevaluatedConst<DbInterner>,
    ) -> Self {
        todo!()
    }

    fn new_expr(
        interner: DbInterner,
        expr: <DbInterner as rustc_type_ir::Interner>::ExprConst,
    ) -> Self {
        todo!()
    }

    fn new_error(
        interner: DbInterner,
        guar: <DbInterner as rustc_type_ir::Interner>::ErrorGuaranteed,
    ) -> Self {
        todo!()
    }
}

impl visit::Flags for Const {
    fn flags(&self) -> rustc_type_ir::TypeFlags {
        todo!()
    }

    fn outer_exclusive_binder(&self) -> rustc_type_ir::DebruijnIndex {
        todo!()
    }
}

impl fold::TypeSuperFoldable<DbInterner> for Const {
    fn try_super_fold_with<F: fold::FallibleTypeFolder<DbInterner>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        todo!()
    }
}

impl visit::TypeSuperVisitable<DbInterner> for Const {
    fn super_visit_with<V: visit::TypeVisitor<DbInterner>>(&self, visitor: &mut V) -> V::Result {
        todo!()
    }
}

impl inherent::IntoKind for Const {
    type Kind = ConstKind<DbInterner>;

    fn kind(self) -> Self::Kind {
        todo!()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Term {
    Ty(Ty),
    Const(Const),
}

impl inherent::Term<DbInterner> for Term {}

todo_structural!(Term);

impl inherent::IntoKind for Term {
    type Kind = TermKind<DbInterner>;

    fn kind(self) -> Self::Kind {
        match self {
            Self::Ty(ty) => TermKind::Ty(ty),
            Self::Const(ct) => TermKind::Const(ct),
        }
    }
}

impl From<Ty> for Term {
    fn from(value: Ty) -> Self {
        todo!()
    }
}

impl From<Const> for Term {
    fn from(value: Const) -> Self {
        todo!()
    }
}

impl<T> ir_print::IrPrint<T> for DbInterner {
    fn print(t: &T, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
    }

    fn print_debug(t: &T, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct BoundVarKinds(Vec<BoundVarKind>);

todo_structural!(BoundVarKinds);

impl BoundVarKinds {
    pub fn new(data: impl IntoIterator<Item = BoundVarKind>) -> Self {
        BoundVarKinds(data.into_iter().collect())
    }
}

impl Default for BoundVarKinds {
    fn default() -> Self {
        todo!()
    }
}

pub struct BoundVarKindsIter;
impl Iterator for BoundVarKindsIter {
    type Item = BoundVarKind;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl inherent::SliceLike for BoundVarKinds {
    type Item = BoundVarKind;
    type IntoIter = BoundVarKindsIter;

    fn iter(self) -> Self::IntoIter {
        todo!()
    }

    fn as_slice(&self) -> &[Self::Item] {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum BoundTyKind {
    Anon,
    Param(GenericDefId),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum BoundRegionKind {
    Anon,
    Named(GenericDefId),
    ClosureEnv,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum BoundVarKind {
    Ty(BoundTyKind),
    Region(BoundRegionKind),
    Const,
}

todo_structural!(BoundVarKind);

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct PredefinedOpaques;

todo_structural!(PredefinedOpaques);

impl std::ops::Deref for PredefinedOpaques {
    type Target = PredefinedOpaquesData<DbInterner>;

    fn deref(&self) -> &Self::Target {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct DefiningOpaqueTypes;

todo_structural!(DefiningOpaqueTypes);

impl Default for DefiningOpaqueTypes {
    fn default() -> Self {
        todo!()
    }
}

pub struct DefiningOpaqueTypesIter;
impl Iterator for DefiningOpaqueTypesIter {
    type Item = GenericDefId;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl inherent::SliceLike for DefiningOpaqueTypes {
    type Item = GenericDefId;
    type IntoIter = DefiningOpaqueTypesIter;

    fn iter(self) -> Self::IntoIter {
        todo!()
    }

    fn as_slice(&self) -> &[Self::Item] {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct CanonicalVars;

todo_structural!(CanonicalVars);

impl Default for CanonicalVars {
    fn default() -> Self {
        todo!()
    }
}

pub struct CanonicalVarsIter;
impl Iterator for CanonicalVarsIter {
    type Item = CanonicalVarInfo<DbInterner>;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl inherent::SliceLike for CanonicalVars {
    type Item = CanonicalVarInfo<DbInterner>;
    type IntoIter = CanonicalVarsIter;

    fn iter(self) -> Self::IntoIter {
        todo!()
    }

    fn as_slice(&self) -> &[Self::Item] {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ExternalConstraints;

todo_structural!(ExternalConstraints);

impl std::ops::Deref for ExternalConstraints {
    type Target = ExternalConstraintsData<DbInterner>;

    fn deref(&self) -> &Self::Target {
        todo!()
    }
}

pub struct DepNodeIndex;

#[derive(Debug)]
pub struct Tracked<T: fmt::Debug + Clone>(T);

type InternedTys = Interned<InternedWrapper<SmallVec<[Ty; 2]>>>;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Tys(InternedTys);

todo_structural!(Tys);

impl Tys {
    pub fn new(data: impl IntoIterator<Item = Ty>) -> Self {
        Tys(Interned::new(InternedWrapper(data.into_iter().collect())))
    }
}

impl Default for Tys {
    fn default() -> Self {
        todo!()
    }
}

pub struct TysIter;
impl Iterator for TysIter {
    type Item = Ty;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl inherent::SliceLike for Tys {
    type Item = Ty;
    type IntoIter = TysIter;

    fn iter(self) -> Self::IntoIter {
        todo!()
    }

    fn as_slice(&self) -> &[Self::Item] {
        todo!()
    }
}

impl inherent::Tys<DbInterner> for Tys {
    fn inputs(self) -> <DbInterner as rustc_type_ir::Interner>::FnInputTys {
        todo!()
    }

    fn output(self) -> <DbInterner as rustc_type_ir::Interner>::Ty {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct FnInputTys;

todo_structural!(FnInputTys);

impl Default for FnInputTys {
    fn default() -> Self {
        todo!()
    }
}

pub struct FnInputTysIter;
impl Iterator for FnInputTysIter {
    type Item = Ty;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl inherent::SliceLike for FnInputTys {
    type Item = Ty;
    type IntoIter = FnInputTysIter;

    fn iter(self) -> Self::IntoIter {
        todo!()
    }

    fn as_slice(&self) -> &[Self::Item] {
        todo!()
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct ParamTy {
    pub(crate) index: u32,
}

todo_structural!(ParamTy);

impl inherent::ParamLike for ParamTy {
    fn index(&self) -> u32 {
        self.index
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct BoundTy(rustc_type_ir::BoundVar);

todo_structural!(BoundTy);

impl BoundTy {
    pub fn new(var: rustc_type_ir::BoundVar) -> Self {
        BoundTy(var)
    }
}

impl inherent::BoundVarLike<DbInterner> for BoundTy {
    fn var(self) -> rustc_type_ir::BoundVar {
        self.0
    }

    fn assert_eq(self, var: <DbInterner as rustc_type_ir::Interner>::BoundVarKind) {
        todo!()
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct PlaceholderTy {
    universe: rustc_type_ir::UniverseIndex,
    var: rustc_type_ir::BoundVar,
}

todo_structural!(PlaceholderTy);

impl inherent::PlaceholderLike for PlaceholderTy {
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
pub struct ErrorGuaranteed;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct BoundExistentialPredicates;

todo_structural!(BoundExistentialPredicates);

pub struct BoundExistentialPredicatesIter;
impl Iterator for BoundExistentialPredicatesIter {
    type Item = rustc_type_ir::Binder<DbInterner, rustc_type_ir::ExistentialPredicate<DbInterner>>;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl inherent::SliceLike for BoundExistentialPredicates {
    type Item = rustc_type_ir::Binder<DbInterner, rustc_type_ir::ExistentialPredicate<DbInterner>>;
    type IntoIter = BoundExistentialPredicatesIter;

    fn iter(self) -> Self::IntoIter {
        todo!()
    }

    fn as_slice(&self) -> &[Self::Item] {
        todo!()
    }
}

impl inherent::BoundExistentialPredicates<DbInterner> for BoundExistentialPredicates {
    fn principal_def_id(&self) -> Option<<DbInterner as rustc_type_ir::Interner>::DefId> {
        todo!()
    }

    fn principal(
        self,
    ) -> Option<rustc_type_ir::Binder<DbInterner, rustc_type_ir::ExistentialTraitRef<DbInterner>>>
    {
        todo!()
    }

    fn auto_traits(
        self,
    ) -> impl IntoIterator<Item = <DbInterner as rustc_type_ir::Interner>::DefId> {
        todo!();
        None
    }

    fn projection_bounds(
        self,
    ) -> impl IntoIterator<
        Item = rustc_type_ir::Binder<DbInterner, rustc_type_ir::ExistentialProjection<DbInterner>>,
    > {
        todo!();
        None
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct AllocId;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct Pat;

todo_structural!(Pat);

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct Safety(chalk_ir::Safety);

impl Safety {
    pub fn new(safety: chalk_ir::Safety) -> Self {
        Safety(safety)
    }
}

todo_structural!(Safety);

impl inherent::Safety<DbInterner> for Safety {
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
pub struct Abi(FnAbi);

todo_structural!(Abi);

impl Abi {
    pub fn new(abi: FnAbi) -> Self {
        Abi(abi)
    }
}

impl inherent::Abi<DbInterner> for Abi {
    fn rust() -> Self {
        todo!()
    }

    fn is_rust(self) -> bool {
        todo!()
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct PlaceholderConst;

todo_structural!(PlaceholderConst);

impl inherent::PlaceholderLike for PlaceholderConst {
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
pub struct ParamConst {
    pub index: u32,
}

todo_structural!(ParamConst);

impl inherent::ParamLike for ParamConst {
    fn index(&self) -> u32 {
        self.index
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct BoundConst(rustc_type_ir::BoundVar);

impl BoundConst {
    pub fn new(var: rustc_type_ir::BoundVar) -> Self {
        BoundConst(var)
    }
}

todo_structural!(BoundConst);

impl inherent::BoundVarLike<DbInterner> for BoundConst {
    fn var(self) -> rustc_type_ir::BoundVar {
        self.0
    }

    fn assert_eq(self, var: <DbInterner as rustc_type_ir::Interner>::BoundVarKind) {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ValueConst(ConstScalar);

impl ValueConst {
    pub fn new(scalar: ConstScalar) -> Self {
        ValueConst(scalar)
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct ExprConst;

todo_structural!(ExprConst);

impl inherent::ExprConst<DbInterner> for ExprConst {
    fn args(self) -> <DbInterner as rustc_type_ir::Interner>::GenericArgs {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Region(Interned<InternedWrapper<rustc_type_ir::RegionKind<DbInterner>>>);

impl Region {
    pub fn new(kind: rustc_type_ir::RegionKind<DbInterner>) -> Self {
        Region(Interned::new(InternedWrapper(kind)))
    }
}

impl PartialOrd for Region {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        todo!()
    }
}

impl Ord for Region {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        todo!()
    }
}

todo_structural!(Region);

impl inherent::Region<DbInterner> for Region {
    fn new_bound(
        interner: DbInterner,
        debruijn: rustc_type_ir::DebruijnIndex,
        var: <DbInterner as rustc_type_ir::Interner>::BoundRegion,
    ) -> Self {
        todo!()
    }

    fn new_anon_bound(
        interner: DbInterner,
        debruijn: rustc_type_ir::DebruijnIndex,
        var: rustc_type_ir::BoundVar,
    ) -> Self {
        todo!()
    }

    fn new_static(interner: DbInterner) -> Self {
        todo!()
    }
}

impl visit::Flags for Region {
    fn flags(&self) -> rustc_type_ir::TypeFlags {
        todo!()
    }

    fn outer_exclusive_binder(&self) -> rustc_type_ir::DebruijnIndex {
        todo!()
    }
}

impl fold::TypeSuperFoldable<DbInterner> for Region {
    fn try_super_fold_with<F: fold::FallibleTypeFolder<DbInterner>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        todo!()
    }
}

impl visit::TypeSuperVisitable<DbInterner> for Region {
    fn super_visit_with<V: visit::TypeVisitor<DbInterner>>(&self, visitor: &mut V) -> V::Result {
        todo!()
    }
}

impl inherent::IntoKind for Region {
    type Kind = rustc_type_ir::RegionKind<DbInterner>;

    fn kind(self) -> Self::Kind {
        todo!()
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct EarlyParamRegion {
    pub index: u32,
}

impl inherent::ParamLike for EarlyParamRegion {
    fn index(&self) -> u32 {
        self.index
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct LateParamRegion;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct BoundRegion(rustc_type_ir::BoundVar);

impl BoundRegion {
    pub fn new(var: rustc_type_ir::BoundVar) -> Self {
        BoundRegion(var)
    }
}

todo_structural!(BoundRegion);

impl inherent::BoundVarLike<DbInterner> for BoundRegion {
    fn var(self) -> rustc_type_ir::BoundVar {
        self.0
    }

    fn assert_eq(self, var: <DbInterner as rustc_type_ir::Interner>::BoundVarKind) {
        todo!()
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct PlaceholderRegion {
    universe: rustc_type_ir::UniverseIndex,
    var: rustc_type_ir::BoundVar,
}

todo_structural!(PlaceholderRegion);

impl inherent::PlaceholderLike for PlaceholderRegion {
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
pub struct ParamEnv;

todo_structural!(ParamEnv);

impl inherent::ParamEnv<DbInterner> for ParamEnv {
    fn reveal(&self) -> rustc_type_ir::solve::Reveal {
        todo!()
    }

    fn caller_bounds(
        self,
    ) -> impl IntoIterator<Item = <DbInterner as rustc_type_ir::Interner>::Clause> {
        todo!();
        None
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Predicate;

todo_structural!(Predicate);

impl inherent::Predicate<DbInterner> for Predicate {
    fn as_clause(self) -> Option<<DbInterner as rustc_type_ir::Interner>::Clause> {
        todo!()
    }

    fn is_coinductive(&self, interner: DbInterner) -> bool {
        todo!()
    }

    fn allow_normalization(&self) -> bool {
        todo!()
    }
}

impl elaborate::Elaboratable<DbInterner> for Predicate {
    fn predicate_kind(
        self,
    ) -> rustc_type_ir::Binder<DbInterner, rustc_type_ir::PredicateKind<DbInterner>> {
        todo!()
    }

    fn as_clause(self) -> Option<<DbInterner as rustc_type_ir::Interner>::Clause> {
        todo!()
    }

    fn child(&self, clause: <DbInterner as rustc_type_ir::Interner>::Clause) -> Self {
        todo!()
    }

    fn child_with_derived_cause(
        &self,
        clause: <DbInterner as rustc_type_ir::Interner>::Clause,
        span: <DbInterner as rustc_type_ir::Interner>::Span,
        parent_trait_pred: rustc_type_ir::Binder<
            DbInterner,
            rustc_type_ir::TraitPredicate<DbInterner>,
        >,
        index: usize,
    ) -> Self {
        todo!()
    }
}

impl inherent::IntoKind for Predicate {
    type Kind = rustc_type_ir::Binder<DbInterner, rustc_type_ir::PredicateKind<DbInterner>>;

    fn kind(self) -> Self::Kind {
        todo!()
    }
}

impl visit::Flags for Predicate {
    fn flags(&self) -> rustc_type_ir::TypeFlags {
        todo!()
    }

    fn outer_exclusive_binder(&self) -> rustc_type_ir::DebruijnIndex {
        todo!()
    }
}

impl fold::TypeSuperFoldable<DbInterner> for Predicate {
    fn try_super_fold_with<F: fold::FallibleTypeFolder<DbInterner>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        todo!()
    }
}

impl visit::TypeSuperVisitable<DbInterner> for Predicate {
    fn super_visit_with<V: visit::TypeVisitor<DbInterner>>(&self, visitor: &mut V) -> V::Result {
        todo!()
    }
}

impl rustc_type_ir::UpcastFrom<DbInterner, rustc_type_ir::PredicateKind<DbInterner>> for Predicate {
    fn upcast_from(from: rustc_type_ir::PredicateKind<DbInterner>, interner: DbInterner) -> Self {
        todo!()
    }
}

impl
    rustc_type_ir::UpcastFrom<
        DbInterner,
        rustc_type_ir::Binder<DbInterner, rustc_type_ir::PredicateKind<DbInterner>>,
    > for Predicate
{
    fn upcast_from(
        from: rustc_type_ir::Binder<DbInterner, rustc_type_ir::PredicateKind<DbInterner>>,
        interner: DbInterner,
    ) -> Self {
        todo!()
    }
}

impl rustc_type_ir::UpcastFrom<DbInterner, rustc_type_ir::ClauseKind<DbInterner>> for Predicate {
    fn upcast_from(from: rustc_type_ir::ClauseKind<DbInterner>, interner: DbInterner) -> Self {
        todo!()
    }
}

impl
    rustc_type_ir::UpcastFrom<
        DbInterner,
        rustc_type_ir::Binder<DbInterner, rustc_type_ir::ClauseKind<DbInterner>>,
    > for Predicate
{
    fn upcast_from(
        from: rustc_type_ir::Binder<DbInterner, rustc_type_ir::ClauseKind<DbInterner>>,
        interner: DbInterner,
    ) -> Self {
        todo!()
    }
}

impl rustc_type_ir::UpcastFrom<DbInterner, Clause> for Predicate {
    fn upcast_from(from: Clause, interner: DbInterner) -> Self {
        todo!()
    }
}

impl rustc_type_ir::UpcastFrom<DbInterner, rustc_type_ir::NormalizesTo<DbInterner>> for Predicate {
    fn upcast_from(from: rustc_type_ir::NormalizesTo<DbInterner>, interner: DbInterner) -> Self {
        todo!()
    }
}

impl rustc_type_ir::UpcastFrom<DbInterner, rustc_type_ir::TraitRef<DbInterner>> for Predicate {
    fn upcast_from(from: rustc_type_ir::TraitRef<DbInterner>, interner: DbInterner) -> Self {
        todo!()
    }
}

impl
    rustc_type_ir::UpcastFrom<
        DbInterner,
        rustc_type_ir::Binder<DbInterner, rustc_type_ir::TraitRef<DbInterner>>,
    > for Predicate
{
    fn upcast_from(
        from: rustc_type_ir::Binder<DbInterner, rustc_type_ir::TraitRef<DbInterner>>,
        interner: DbInterner,
    ) -> Self {
        todo!()
    }
}

impl rustc_type_ir::UpcastFrom<DbInterner, rustc_type_ir::TraitPredicate<DbInterner>>
    for Predicate
{
    fn upcast_from(from: rustc_type_ir::TraitPredicate<DbInterner>, interner: DbInterner) -> Self {
        todo!()
    }
}

impl rustc_type_ir::UpcastFrom<DbInterner, rustc_type_ir::OutlivesPredicate<DbInterner, Ty>>
    for Predicate
{
    fn upcast_from(
        from: rustc_type_ir::OutlivesPredicate<DbInterner, Ty>,
        interner: DbInterner,
    ) -> Self {
        todo!()
    }
}

impl rustc_type_ir::UpcastFrom<DbInterner, rustc_type_ir::OutlivesPredicate<DbInterner, Region>>
    for Predicate
{
    fn upcast_from(
        from: rustc_type_ir::OutlivesPredicate<DbInterner, Region>,
        interner: DbInterner,
    ) -> Self {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Clause;

todo_structural!(Clause);

impl inherent::Clause<DbInterner> for Clause {
    fn as_predicate(self) -> <DbInterner as rustc_type_ir::Interner>::Predicate {
        todo!()
    }

    fn instantiate_supertrait(
        self,
        cx: DbInterner,
        trait_ref: rustc_type_ir::Binder<DbInterner, rustc_type_ir::TraitRef<DbInterner>>,
    ) -> Self {
        todo!()
    }
}

impl elaborate::Elaboratable<DbInterner> for Clause {
    fn predicate_kind(
        self,
    ) -> rustc_type_ir::Binder<DbInterner, rustc_type_ir::PredicateKind<DbInterner>> {
        todo!()
    }

    fn as_clause(self) -> Option<<DbInterner as rustc_type_ir::Interner>::Clause> {
        todo!()
    }

    fn child(&self, clause: <DbInterner as rustc_type_ir::Interner>::Clause) -> Self {
        todo!()
    }

    fn child_with_derived_cause(
        &self,
        clause: <DbInterner as rustc_type_ir::Interner>::Clause,
        span: <DbInterner as rustc_type_ir::Interner>::Span,
        parent_trait_pred: rustc_type_ir::Binder<
            DbInterner,
            rustc_type_ir::TraitPredicate<DbInterner>,
        >,
        index: usize,
    ) -> Self {
        todo!()
    }
}

impl inherent::IntoKind for Clause {
    type Kind = rustc_type_ir::Binder<DbInterner, rustc_type_ir::ClauseKind<DbInterner>>;

    fn kind(self) -> Self::Kind {
        todo!()
    }
}

impl visit::Flags for Clause {
    fn flags(&self) -> rustc_type_ir::TypeFlags {
        todo!()
    }

    fn outer_exclusive_binder(&self) -> rustc_type_ir::DebruijnIndex {
        todo!()
    }
}

impl fold::TypeSuperFoldable<DbInterner> for Clause {
    fn try_super_fold_with<F: fold::FallibleTypeFolder<DbInterner>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        todo!()
    }
}

impl visit::TypeSuperVisitable<DbInterner> for Clause {
    fn super_visit_with<V: visit::TypeVisitor<DbInterner>>(&self, visitor: &mut V) -> V::Result {
        todo!()
    }
}

impl
    rustc_type_ir::UpcastFrom<
        DbInterner,
        rustc_type_ir::Binder<DbInterner, rustc_type_ir::ClauseKind<DbInterner>>,
    > for Clause
{
    fn upcast_from(
        from: rustc_type_ir::Binder<DbInterner, rustc_type_ir::ClauseKind<DbInterner>>,
        interner: DbInterner,
    ) -> Self {
        todo!()
    }
}

impl rustc_type_ir::UpcastFrom<DbInterner, rustc_type_ir::TraitRef<DbInterner>> for Clause {
    fn upcast_from(from: rustc_type_ir::TraitRef<DbInterner>, interner: DbInterner) -> Self {
        todo!()
    }
}

impl
    rustc_type_ir::UpcastFrom<
        DbInterner,
        rustc_type_ir::Binder<DbInterner, rustc_type_ir::TraitRef<DbInterner>>,
    > for Clause
{
    fn upcast_from(
        from: rustc_type_ir::Binder<DbInterner, rustc_type_ir::TraitRef<DbInterner>>,
        interner: DbInterner,
    ) -> Self {
        todo!()
    }
}

impl rustc_type_ir::UpcastFrom<DbInterner, rustc_type_ir::TraitPredicate<DbInterner>> for Clause {
    fn upcast_from(from: rustc_type_ir::TraitPredicate<DbInterner>, interner: DbInterner) -> Self {
        todo!()
    }
}

impl
    rustc_type_ir::UpcastFrom<
        DbInterner,
        rustc_type_ir::Binder<DbInterner, rustc_type_ir::TraitPredicate<DbInterner>>,
    > for Clause
{
    fn upcast_from(
        from: rustc_type_ir::Binder<DbInterner, rustc_type_ir::TraitPredicate<DbInterner>>,
        interner: DbInterner,
    ) -> Self {
        todo!()
    }
}

impl rustc_type_ir::UpcastFrom<DbInterner, rustc_type_ir::ProjectionPredicate<DbInterner>>
    for Clause
{
    fn upcast_from(
        from: rustc_type_ir::ProjectionPredicate<DbInterner>,
        interner: DbInterner,
    ) -> Self {
        todo!()
    }
}

impl
    rustc_type_ir::UpcastFrom<
        DbInterner,
        rustc_type_ir::Binder<DbInterner, rustc_type_ir::ProjectionPredicate<DbInterner>>,
    > for Clause
{
    fn upcast_from(
        from: rustc_type_ir::Binder<DbInterner, rustc_type_ir::ProjectionPredicate<DbInterner>>,
        interner: DbInterner,
    ) -> Self {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Clauses;

todo_structural!(Clauses);

pub struct ClausesIter;
impl Iterator for ClausesIter {
    type Item = ();

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl inherent::SliceLike for Clauses {
    type Item = ();
    type IntoIter = ClausesIter;

    fn iter(self) -> Self::IntoIter {
        todo!()
    }

    fn as_slice(&self) -> &[Self::Item] {
        todo!()
    }
}

impl visit::Flags for Clauses {
    fn flags(&self) -> rustc_type_ir::TypeFlags {
        todo!()
    }

    fn outer_exclusive_binder(&self) -> rustc_type_ir::DebruijnIndex {
        todo!()
    }
}

impl fold::TypeSuperFoldable<DbInterner> for Clauses {
    fn try_super_fold_with<F: fold::FallibleTypeFolder<DbInterner>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        todo!()
    }
}

impl visit::TypeSuperVisitable<DbInterner> for Clauses {
    fn super_visit_with<V: visit::TypeVisitor<DbInterner>>(&self, visitor: &mut V) -> V::Result {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct GenericsOf;

todo_structural!(GenericsOf);

impl inherent::GenericsOf<DbInterner> for GenericsOf {
    fn count(&self) -> usize {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct VariancesOf;

todo_structural!(VariancesOf);

pub struct VariancesOfIter;
impl Iterator for VariancesOfIter {
    type Item = rustc_type_ir::Variance;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl inherent::SliceLike for VariancesOf {
    type Item = rustc_type_ir::Variance;
    type IntoIter = VariancesOfIter;

    fn iter(self) -> Self::IntoIter {
        todo!()
    }

    fn as_slice(&self) -> &[Self::Item] {
        todo!()
    }
}

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

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ReprOptions;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct AdtDefData {
    pub did: GenericDefId,
    pub id: AdtId,
    pub variants: Vec<(VariantIdx, VariantDef)>,
    pub flags: AdtFlags,
    pub repr: ReprOptions,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct AdtDef(AdtDefData);

impl AdtDef {
    pub fn new(def_id: AdtId) -> Self {
        todo!()
    }
}

todo_structural!(AdtDef);

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
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Features;

impl inherent::Features<DbInterner> for Features {
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
pub struct UnsizingParams;

impl std::ops::Deref for UnsizingParams {
    type Target = BitSet<u32>;

    fn deref(&self) -> &Self::Target {
        todo!()
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
        todo!()
    }

    type DefiningOpaqueTypes = DefiningOpaqueTypes;

    type CanonicalVars = CanonicalVars;

    fn mk_canonical_var_infos(
        self,
        infos: &[rustc_type_ir::CanonicalVarInfo<Self>],
    ) -> Self::CanonicalVars {
        todo!()
    }

    type ExternalConstraints = ExternalConstraints;

    fn mk_external_constraints(
        self,
        data: rustc_type_ir::solve::ExternalConstraintsData<Self>,
    ) -> Self::ExternalConstraints {
        todo!()
    }

    type DepNodeIndex = DepNodeIndex;

    type Tracked<T: fmt::Debug + Clone> = Tracked<T>;

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

    type Ty = Ty;
    type Tys = Tys;
    type FnInputTys = FnInputTys;
    type ParamTy = ParamTy;
    type BoundTy = BoundTy;
    type PlaceholderTy = PlaceholderTy;

    type ErrorGuaranteed = ErrorGuaranteed;
    type BoundExistentialPredicates = BoundExistentialPredicates;
    type AllocId = AllocId;
    type Pat = Pat;
    type Safety = Safety;
    type Abi = Abi;

    type Const = Const;
    type PlaceholderConst = PlaceholderConst;
    type ParamConst = ParamConst;
    type BoundConst = BoundConst;
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
        todo!()
    }

    fn evaluation_is_concurrent(&self) -> bool {
        todo!()
    }

    fn expand_abstract_consts<T: rustc_type_ir::fold::TypeFoldable<Self>>(self, t: T) -> T {
        todo!()
    }

    type GenericsOf = GenericsOf;

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

    type Features = Features;

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
