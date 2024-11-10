#![allow(unused)]

use base_db::{ra_salsa::InternId, CrateId};
use chalk_ir::{ProgramClauseImplication, SeparatorTraitRef, Variance};
use hir_def::{BlockId, TypeAliasId};
use intern::{impl_internable, InternStorage, Internable, Interned};
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

use crate::db::HirDatabase;

#[derive(Copy, Clone)]
pub struct RustcInterner<'a> {
    pub(crate) db: &'a dyn HirDatabase,
    pub(crate) krate: CrateId,
    pub(crate) block: Option<BlockId>,
}

macro_rules! todo_structural {
    ($t:ty) => {
        impl<'cx> relate::Relate<RustcInterner<'cx>> for $t {
            fn relate<R: relate::TypeRelation<RustcInterner<'cx>>>(
                _relation: &mut R,
                _a: Self,
                _b: Self,
            ) -> relate::RelateResult<RustcInterner<'cx>, Self> {
                todo!()
            }
        }

        impl<'cx> fold::TypeFoldable<RustcInterner<'cx>> for $t {
            fn try_fold_with<F: fold::FallibleTypeFolder<RustcInterner<'cx>>>(
                self,
                _folder: &mut F,
            ) -> Result<Self, F::Error> {
                todo!()
            }
        }

        impl<'cx> visit::TypeVisitable<RustcInterner<'cx>> for $t {
            fn visit_with<V: visit::TypeVisitor<RustcInterner<'cx>>>(
                &self,
                _visitor: &mut V,
            ) -> V::Result {
                todo!()
            }
        }
    };
}

impl inherent::DefId<RustcInterner<'_>> for InternId {
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

impl inherent::Span<RustcInterner<'_>> for RustcSpan {
    fn dummy() -> Self {
        RustcSpan(None)
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct GenericArgs<'cx>(&'cx ());

impl<'cx> inherent::GenericArgs<RustcInterner<'cx>> for GenericArgs<'cx> {
    fn dummy() -> Self {
        todo!()
    }

    fn rebase_onto(
        self,
        interner: RustcInterner<'cx>,
        source_def_id: <RustcInterner<'cx> as rustc_type_ir::Interner>::DefId,
        target: <RustcInterner<'cx> as rustc_type_ir::Interner>::GenericArgs,
    ) -> <RustcInterner<'cx> as rustc_type_ir::Interner>::GenericArgs {
        todo!()
    }

    fn type_at(self, i: usize) -> <RustcInterner<'cx> as rustc_type_ir::Interner>::Ty {
        todo!()
    }

    fn region_at(self, i: usize) -> <RustcInterner<'cx> as rustc_type_ir::Interner>::Region {
        todo!()
    }

    fn const_at(self, i: usize) -> <RustcInterner<'cx> as rustc_type_ir::Interner>::Const {
        todo!()
    }

    fn identity_for_item(
        interner: RustcInterner<'cx>,
        def_id: <RustcInterner<'cx> as rustc_type_ir::Interner>::DefId,
    ) -> <RustcInterner<'cx> as rustc_type_ir::Interner>::GenericArgs {
        todo!()
    }

    fn extend_with_error(
        interner: RustcInterner<'cx>,
        def_id: <RustcInterner<'cx> as rustc_type_ir::Interner>::DefId,
        original_args: &[RustcGenericArg<'cx>],
    ) -> <RustcInterner<'cx> as rustc_type_ir::Interner>::GenericArgs {
        todo!()
    }

    fn split_closure_args(self) -> rustc_type_ir::ClosureArgsParts<RustcInterner<'cx>> {
        todo!()
    }

    fn split_coroutine_closure_args(
        self,
    ) -> rustc_type_ir::CoroutineClosureArgsParts<RustcInterner<'cx>> {
        todo!()
    }

    fn split_coroutine_args(self) -> rustc_type_ir::CoroutineArgsParts<RustcInterner<'cx>> {
        todo!()
    }
}

todo_structural!(GenericArgs<'cx>);

pub struct GenericArgsIter<'cx>(&'cx ());
impl<'cx> Iterator for GenericArgsIter<'cx> {
    type Item = RustcGenericArg<'cx>;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl<'cx> inherent::SliceLike for GenericArgs<'cx> {
    type Item = RustcGenericArg<'cx>;
    type IntoIter = GenericArgsIter<'cx>;

    fn iter(self) -> Self::IntoIter {
        todo!()
    }

    fn as_slice(&self) -> &[Self::Item] {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RustcGenericArg<'cx>(&'cx ());

todo_structural!(RustcGenericArg<'cx>);

impl<'cx> inherent::GenericArg<RustcInterner<'cx>> for RustcGenericArg<'cx> {}

impl<'cx> inherent::IntoKind for RustcGenericArg<'cx> {
    type Kind = GenericArgKind<RustcInterner<'cx>>;

    fn kind(self) -> Self::Kind {
        todo!()
    }
}

impl<'cx> From<RustcTy<'cx>> for RustcGenericArg<'cx> {
    fn from(value: RustcTy<'cx>) -> Self {
        todo!()
    }
}

impl<'cx> From<RustcConst<'cx>> for RustcGenericArg<'cx> {
    fn from(value: RustcConst<'cx>) -> Self {
        todo!()
    }
}

impl<'cx> From<RustcRegion<'cx>> for RustcGenericArg<'cx> {
    fn from(value: RustcRegion<'cx>) -> Self {
        todo!()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RustcTy<'cx>(&'cx ());

todo_structural!(RustcTy<'cx>);

impl<'cx> inherent::Ty<RustcInterner<'cx>> for RustcTy<'cx> {
    fn new_unit(interner: RustcInterner<'cx>) -> Self {
        todo!()
    }

    fn new_bool(interner: RustcInterner<'cx>) -> Self {
        todo!()
    }

    fn new_u8(interner: RustcInterner<'cx>) -> Self {
        todo!()
    }

    fn new_usize(interner: RustcInterner<'cx>) -> Self {
        todo!()
    }

    fn new_infer(interner: RustcInterner<'cx>, var: rustc_type_ir::InferTy) -> Self {
        todo!()
    }

    fn new_var(interner: RustcInterner<'cx>, var: rustc_type_ir::TyVid) -> Self {
        todo!()
    }

    fn new_param(
        interner: RustcInterner<'cx>,
        param: <RustcInterner<'cx> as rustc_type_ir::Interner>::ParamTy,
    ) -> Self {
        todo!()
    }

    fn new_placeholder(
        interner: RustcInterner<'cx>,
        param: <RustcInterner<'cx> as rustc_type_ir::Interner>::PlaceholderTy,
    ) -> Self {
        todo!()
    }

    fn new_bound(
        interner: RustcInterner<'cx>,
        debruijn: rustc_type_ir::DebruijnIndex,
        var: <RustcInterner<'cx> as rustc_type_ir::Interner>::BoundTy,
    ) -> Self {
        todo!()
    }

    fn new_anon_bound(
        interner: RustcInterner<'cx>,
        debruijn: rustc_type_ir::DebruijnIndex,
        var: rustc_type_ir::BoundVar,
    ) -> Self {
        todo!()
    }

    fn new_alias(
        interner: RustcInterner<'cx>,
        kind: rustc_type_ir::AliasTyKind,
        alias_ty: rustc_type_ir::AliasTy<RustcInterner<'cx>>,
    ) -> Self {
        todo!()
    }

    fn new_error(
        interner: RustcInterner<'cx>,
        guar: <RustcInterner<'cx> as rustc_type_ir::Interner>::ErrorGuaranteed,
    ) -> Self {
        todo!()
    }

    fn new_adt(
        interner: RustcInterner<'cx>,
        adt_def: <RustcInterner<'cx> as rustc_type_ir::Interner>::AdtDef,
        args: <RustcInterner<'cx> as rustc_type_ir::Interner>::GenericArgs,
    ) -> Self {
        todo!()
    }

    fn new_foreign(
        interner: RustcInterner<'cx>,
        def_id: <RustcInterner<'cx> as rustc_type_ir::Interner>::DefId,
    ) -> Self {
        todo!()
    }

    fn new_dynamic(
        interner: RustcInterner<'cx>,
        preds: <RustcInterner<'cx> as rustc_type_ir::Interner>::BoundExistentialPredicates,
        region: <RustcInterner<'cx> as rustc_type_ir::Interner>::Region,
        kind: rustc_type_ir::DynKind,
    ) -> Self {
        todo!()
    }

    fn new_coroutine(
        interner: RustcInterner<'cx>,
        def_id: <RustcInterner<'cx> as rustc_type_ir::Interner>::DefId,
        args: <RustcInterner<'cx> as rustc_type_ir::Interner>::GenericArgs,
    ) -> Self {
        todo!()
    }

    fn new_coroutine_closure(
        interner: RustcInterner<'cx>,
        def_id: <RustcInterner<'cx> as rustc_type_ir::Interner>::DefId,
        args: <RustcInterner<'cx> as rustc_type_ir::Interner>::GenericArgs,
    ) -> Self {
        todo!()
    }

    fn new_closure(
        interner: RustcInterner<'cx>,
        def_id: <RustcInterner<'cx> as rustc_type_ir::Interner>::DefId,
        args: <RustcInterner<'cx> as rustc_type_ir::Interner>::GenericArgs,
    ) -> Self {
        todo!()
    }

    fn new_coroutine_witness(
        interner: RustcInterner<'cx>,
        def_id: <RustcInterner<'cx> as rustc_type_ir::Interner>::DefId,
        args: <RustcInterner<'cx> as rustc_type_ir::Interner>::GenericArgs,
    ) -> Self {
        todo!()
    }

    fn new_ptr(interner: RustcInterner<'cx>, ty: Self, mutbl: rustc_ast_ir::Mutability) -> Self {
        todo!()
    }

    fn new_ref(
        interner: RustcInterner<'cx>,
        region: <RustcInterner<'cx> as rustc_type_ir::Interner>::Region,
        ty: Self,
        mutbl: rustc_ast_ir::Mutability,
    ) -> Self {
        todo!()
    }

    fn new_array_with_const_len(
        interner: RustcInterner<'cx>,
        ty: Self,
        len: <RustcInterner<'cx> as rustc_type_ir::Interner>::Const,
    ) -> Self {
        todo!()
    }

    fn new_slice(interner: RustcInterner<'cx>, ty: Self) -> Self {
        todo!()
    }

    fn new_tup(
        interner: RustcInterner<'cx>,
        tys: &[<RustcInterner<'cx> as rustc_type_ir::Interner>::Ty],
    ) -> Self {
        todo!()
    }

    fn new_tup_from_iter<It, T>(interner: RustcInterner<'cx>, iter: It) -> T::Output
    where
        It: Iterator<Item = T>,
        T: rustc_type_ir::CollectAndApply<Self, Self>,
    {
        todo!()
    }

    fn new_fn_def(
        interner: RustcInterner<'cx>,
        def_id: <RustcInterner<'cx> as rustc_type_ir::Interner>::DefId,
        args: <RustcInterner<'cx> as rustc_type_ir::Interner>::GenericArgs,
    ) -> Self {
        todo!()
    }

    fn new_fn_ptr(
        interner: RustcInterner<'cx>,
        sig: rustc_type_ir::Binder<RustcInterner<'cx>, rustc_type_ir::FnSig<RustcInterner<'cx>>>,
    ) -> Self {
        todo!()
    }

    fn new_pat(
        interner: RustcInterner<'cx>,
        ty: Self,
        pat: <RustcInterner<'cx> as rustc_type_ir::Interner>::Pat,
    ) -> Self {
        todo!()
    }

    fn tuple_fields(self) -> <RustcInterner<'cx> as rustc_type_ir::Interner>::Tys {
        todo!()
    }

    fn to_opt_closure_kind(self) -> Option<rustc_type_ir::ClosureKind> {
        todo!()
    }

    fn from_closure_kind(interner: RustcInterner<'cx>, kind: rustc_type_ir::ClosureKind) -> Self {
        todo!()
    }

    fn from_coroutine_closure_kind(
        interner: RustcInterner<'cx>,
        kind: rustc_type_ir::ClosureKind,
    ) -> Self {
        todo!()
    }

    fn discriminant_ty(
        self,
        interner: RustcInterner<'cx>,
    ) -> <RustcInterner<'cx> as rustc_type_ir::Interner>::Ty {
        todo!()
    }

    fn async_destructor_ty(
        self,
        interner: RustcInterner<'cx>,
    ) -> <RustcInterner<'cx> as rustc_type_ir::Interner>::Ty {
        todo!()
    }
}

impl<'cx> visit::Flags for RustcTy<'cx> {
    fn flags(&self) -> rustc_type_ir::TypeFlags {
        todo!()
    }

    fn outer_exclusive_binder(&self) -> rustc_type_ir::DebruijnIndex {
        todo!()
    }
}

impl<'cx> fold::TypeSuperFoldable<RustcInterner<'cx>> for RustcTy<'cx> {
    fn try_super_fold_with<F: fold::FallibleTypeFolder<RustcInterner<'cx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        todo!()
    }
}

impl<'cx> visit::TypeSuperVisitable<RustcInterner<'cx>> for RustcTy<'cx> {
    fn super_visit_with<V: visit::TypeVisitor<RustcInterner<'cx>>>(
        &self,
        visitor: &mut V,
    ) -> V::Result {
        todo!()
    }
}

impl<'cx> inherent::IntoKind for RustcTy<'cx> {
    type Kind = rustc_type_ir::TyKind<RustcInterner<'cx>>;

    fn kind(self) -> Self::Kind {
        todo!()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RustcConst<'cx>(&'cx ());

todo_structural!(RustcConst<'cx>);

impl<'cx> inherent::Const<RustcInterner<'cx>> for RustcConst<'cx> {
    fn try_to_target_usize(self, interner: RustcInterner<'cx>) -> Option<u64> {
        todo!()
    }

    fn new_infer(interner: RustcInterner<'cx>, var: rustc_type_ir::InferConst) -> Self {
        todo!()
    }

    fn new_var(interner: RustcInterner<'cx>, var: rustc_type_ir::ConstVid) -> Self {
        todo!()
    }

    fn new_bound(
        interner: RustcInterner<'cx>,
        debruijn: rustc_type_ir::DebruijnIndex,
        var: <RustcInterner<'cx> as rustc_type_ir::Interner>::BoundConst,
    ) -> Self {
        todo!()
    }

    fn new_anon_bound(
        interner: RustcInterner<'cx>,
        debruijn: rustc_type_ir::DebruijnIndex,
        var: rustc_type_ir::BoundVar,
    ) -> Self {
        todo!()
    }

    fn new_unevaluated(
        interner: RustcInterner<'cx>,
        uv: rustc_type_ir::UnevaluatedConst<RustcInterner<'cx>>,
    ) -> Self {
        todo!()
    }

    fn new_expr(
        interner: RustcInterner<'cx>,
        expr: <RustcInterner<'cx> as rustc_type_ir::Interner>::ExprConst,
    ) -> Self {
        todo!()
    }

    fn new_error(
        interner: RustcInterner<'cx>,
        guar: <RustcInterner<'cx> as rustc_type_ir::Interner>::ErrorGuaranteed,
    ) -> Self {
        todo!()
    }
}

impl<'cx> visit::Flags for RustcConst<'cx> {
    fn flags(&self) -> rustc_type_ir::TypeFlags {
        todo!()
    }

    fn outer_exclusive_binder(&self) -> rustc_type_ir::DebruijnIndex {
        todo!()
    }
}

impl<'cx> fold::TypeSuperFoldable<RustcInterner<'cx>> for RustcConst<'cx> {
    fn try_super_fold_with<F: fold::FallibleTypeFolder<RustcInterner<'cx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        todo!()
    }
}

impl<'cx> visit::TypeSuperVisitable<RustcInterner<'cx>> for RustcConst<'cx> {
    fn super_visit_with<V: visit::TypeVisitor<RustcInterner<'cx>>>(
        &self,
        visitor: &mut V,
    ) -> V::Result {
        todo!()
    }
}

impl<'cx> inherent::IntoKind for RustcConst<'cx> {
    type Kind = ConstKind<RustcInterner<'cx>>;

    fn kind(self) -> Self::Kind {
        todo!()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RustcTerm<'cx> {
    Ty(RustcTy<'cx>),
    Const(RustcConst<'cx>),
}

impl<'cx> inherent::Term<RustcInterner<'cx>> for RustcTerm<'cx> {}

todo_structural!(RustcTerm<'cx>);

impl<'cx> inherent::IntoKind for RustcTerm<'cx> {
    type Kind = TermKind<RustcInterner<'cx>>;

    fn kind(self) -> Self::Kind {
        match self {
            Self::Ty(ty) => TermKind::Ty(ty),
            Self::Const(ct) => TermKind::Const(ct),
        }
    }
}

impl<'cx> From<RustcTy<'cx>> for RustcTerm<'cx> {
    fn from(value: RustcTy<'cx>) -> Self {
        todo!()
    }
}

impl<'cx> From<RustcConst<'cx>> for RustcTerm<'cx> {
    fn from(value: RustcConst<'cx>) -> Self {
        todo!()
    }
}

impl<T> ir_print::IrPrint<T> for RustcInterner<'_> {
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
pub struct RustcPredefinedOpaques<'cx>(&'cx ());

todo_structural!(RustcPredefinedOpaques<'cx>);

impl<'cx> std::ops::Deref for RustcPredefinedOpaques<'cx> {
    type Target = PredefinedOpaquesData<RustcInterner<'cx>>;

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
pub struct RustcCanonicalVars<'cx>(&'cx ());

todo_structural!(RustcCanonicalVars<'cx>);

impl<'cx> Default for RustcCanonicalVars<'cx> {
    fn default() -> Self {
        todo!()
    }
}

pub struct RustcCanonicalVarsIter<'cx>(&'cx ());
impl<'cx> Iterator for RustcCanonicalVarsIter<'cx> {
    type Item = CanonicalVarInfo<RustcInterner<'cx>>;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl<'cx> inherent::SliceLike for RustcCanonicalVars<'cx> {
    type Item = CanonicalVarInfo<RustcInterner<'cx>>;
    type IntoIter = RustcCanonicalVarsIter<'cx>;

    fn iter(self) -> Self::IntoIter {
        todo!()
    }

    fn as_slice(&self) -> &[Self::Item] {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RustcExternalConstraints<'cx>(&'cx ());

todo_structural!(RustcExternalConstraints<'cx>);

impl<'cx> std::ops::Deref for RustcExternalConstraints<'cx> {
    type Target = ExternalConstraintsData<RustcInterner<'cx>>;

    fn deref(&self) -> &Self::Target {
        todo!()
    }
}

pub struct RustcDepNodeIndex;

#[derive(Debug)]
pub struct RustcTracked<T: fmt::Debug + Clone>(T);

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RustcTys<'cx>(&'cx ());

todo_structural!(RustcTys<'cx>);

impl<'cx> Default for RustcTys<'cx> {
    fn default() -> Self {
        todo!()
    }
}

pub struct RustcTysIter<'cx>(&'cx ());
impl<'cx> Iterator for RustcTysIter<'cx> {
    type Item = RustcTy<'cx>;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl<'cx> inherent::SliceLike for RustcTys<'cx> {
    type Item = RustcTy<'cx>;
    type IntoIter = RustcTysIter<'cx>;

    fn iter(self) -> Self::IntoIter {
        todo!()
    }

    fn as_slice(&self) -> &[Self::Item] {
        todo!()
    }
}

impl<'cx> inherent::Tys<RustcInterner<'cx>> for RustcTys<'cx> {
    fn inputs(self) -> <RustcInterner<'cx> as rustc_type_ir::Interner>::FnInputTys {
        todo!()
    }

    fn output(self) -> <RustcInterner<'cx> as rustc_type_ir::Interner>::Ty {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RustcFnInputTys<'cx>(&'cx ());

todo_structural!(RustcFnInputTys<'cx>);

impl<'cx> Default for RustcFnInputTys<'cx> {
    fn default() -> Self {
        todo!()
    }
}

pub struct RustcFnInputTysIter<'cx>(&'cx ());
impl<'cx> Iterator for RustcFnInputTysIter<'cx> {
    type Item = RustcTy<'cx>;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl<'cx> inherent::SliceLike for RustcFnInputTys<'cx> {
    type Item = RustcTy<'cx>;
    type IntoIter = RustcFnInputTysIter<'cx>;

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

impl inherent::BoundVarLike<RustcInterner<'_>> for RustcBoundTy {
    fn var(self) -> rustc_type_ir::BoundVar {
        todo!()
    }

    fn assert_eq(self, var: <RustcInterner<'_> as rustc_type_ir::Interner>::BoundVarKind) {
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
pub struct RustcBoundExistentialPredicates<'cx>(&'cx ());

todo_structural!(RustcBoundExistentialPredicates<'cx>);

pub struct RustcBoundExistentialPredicatesIter<'cx>(&'cx ());
impl<'cx> Iterator for RustcBoundExistentialPredicatesIter<'cx> {
    type Item = rustc_type_ir::Binder<
        RustcInterner<'cx>,
        rustc_type_ir::ExistentialPredicate<RustcInterner<'cx>>,
    >;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl<'cx> inherent::SliceLike for RustcBoundExistentialPredicates<'cx> {
    type Item = rustc_type_ir::Binder<
        RustcInterner<'cx>,
        rustc_type_ir::ExistentialPredicate<RustcInterner<'cx>>,
    >;
    type IntoIter = RustcBoundExistentialPredicatesIter<'cx>;

    fn iter(self) -> Self::IntoIter {
        todo!()
    }

    fn as_slice(&self) -> &[Self::Item] {
        todo!()
    }
}

impl<'cx> inherent::BoundExistentialPredicates<RustcInterner<'cx>>
    for RustcBoundExistentialPredicates<'cx>
{
    fn principal_def_id(&self) -> Option<<RustcInterner<'_> as rustc_type_ir::Interner>::DefId> {
        todo!()
    }

    fn principal(
        self,
    ) -> Option<
        rustc_type_ir::Binder<
            RustcInterner<'cx>,
            rustc_type_ir::ExistentialTraitRef<RustcInterner<'cx>>,
        >,
    > {
        todo!()
    }

    fn auto_traits(
        self,
    ) -> impl IntoIterator<Item = <RustcInterner<'cx> as rustc_type_ir::Interner>::DefId> {
        todo!();
        None
    }

    fn projection_bounds(
        self,
    ) -> impl IntoIterator<
        Item = rustc_type_ir::Binder<
            RustcInterner<'cx>,
            rustc_type_ir::ExistentialProjection<RustcInterner<'cx>>,
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

impl inherent::Safety<RustcInterner<'_>> for RustcSafety {
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

impl inherent::Abi<RustcInterner<'_>> for RustcAbi {
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

impl inherent::BoundVarLike<RustcInterner<'_>> for RustcBoundConst {
    fn var(self) -> rustc_type_ir::BoundVar {
        todo!()
    }

    fn assert_eq(self, var: <RustcInterner<'_> as rustc_type_ir::Interner>::BoundVarKind) {
        todo!()
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct RustcValueConst;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct RustcExprConst;

todo_structural!(RustcExprConst);

impl<'cx> inherent::ExprConst<RustcInterner<'cx>> for RustcExprConst {
    fn args(self) -> <RustcInterner<'cx> as rustc_type_ir::Interner>::GenericArgs {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RustcRegion<'cx>(&'cx ());

todo_structural!(RustcRegion<'cx>);

impl<'cx> inherent::Region<RustcInterner<'cx>> for RustcRegion<'cx> {
    fn new_bound(
        interner: RustcInterner<'_>,
        debruijn: rustc_type_ir::DebruijnIndex,
        var: <RustcInterner<'_> as rustc_type_ir::Interner>::BoundRegion,
    ) -> Self {
        todo!()
    }

    fn new_anon_bound(
        interner: RustcInterner<'_>,
        debruijn: rustc_type_ir::DebruijnIndex,
        var: rustc_type_ir::BoundVar,
    ) -> Self {
        todo!()
    }

    fn new_static(interner: RustcInterner<'_>) -> Self {
        todo!()
    }
}

impl<'cx> visit::Flags for RustcRegion<'cx> {
    fn flags(&self) -> rustc_type_ir::TypeFlags {
        todo!()
    }

    fn outer_exclusive_binder(&self) -> rustc_type_ir::DebruijnIndex {
        todo!()
    }
}

impl<'cx> fold::TypeSuperFoldable<RustcInterner<'cx>> for RustcRegion<'cx> {
    fn try_super_fold_with<F: fold::FallibleTypeFolder<RustcInterner<'cx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        todo!()
    }
}

impl<'cx> visit::TypeSuperVisitable<RustcInterner<'cx>> for RustcRegion<'cx> {
    fn super_visit_with<V: visit::TypeVisitor<RustcInterner<'cx>>>(
        &self,
        visitor: &mut V,
    ) -> V::Result {
        todo!()
    }
}

impl<'cx> inherent::IntoKind for RustcRegion<'cx> {
    type Kind = rustc_type_ir::RegionKind<RustcInterner<'cx>>;

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

impl inherent::BoundVarLike<RustcInterner<'_>> for RustcBoundRegion {
    fn var(self) -> rustc_type_ir::BoundVar {
        todo!()
    }

    fn assert_eq(self, var: <RustcInterner<'_> as rustc_type_ir::Interner>::BoundVarKind) {
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

impl<'cx> inherent::ParamEnv<RustcInterner<'cx>> for RustcParamEnv {
    fn reveal(&self) -> rustc_type_ir::solve::Reveal {
        todo!()
    }

    fn caller_bounds(
        self,
    ) -> impl IntoIterator<Item = <RustcInterner<'cx> as rustc_type_ir::Interner>::Clause> {
        todo!();
        None
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RustcPredicate<'cx>(&'cx ());

todo_structural!(RustcPredicate<'cx>);

impl<'cx> inherent::Predicate<RustcInterner<'cx>> for RustcPredicate<'cx> {
    fn as_clause(self) -> Option<<RustcInterner<'cx> as rustc_type_ir::Interner>::Clause> {
        todo!()
    }

    fn is_coinductive(&self, interner: RustcInterner<'cx>) -> bool {
        todo!()
    }

    fn allow_normalization(&self) -> bool {
        todo!()
    }
}

impl<'cx> elaborate::Elaboratable<RustcInterner<'cx>> for RustcPredicate<'cx> {
    fn predicate_kind(
        self,
    ) -> rustc_type_ir::Binder<RustcInterner<'cx>, rustc_type_ir::PredicateKind<RustcInterner<'cx>>>
    {
        todo!()
    }

    fn as_clause(self) -> Option<<RustcInterner<'cx> as rustc_type_ir::Interner>::Clause> {
        todo!()
    }

    fn child(&self, clause: <RustcInterner<'_> as rustc_type_ir::Interner>::Clause) -> Self {
        todo!()
    }

    fn child_with_derived_cause(
        &self,
        clause: <RustcInterner<'cx> as rustc_type_ir::Interner>::Clause,
        span: <RustcInterner<'cx> as rustc_type_ir::Interner>::Span,
        parent_trait_pred: rustc_type_ir::Binder<
            RustcInterner<'cx>,
            rustc_type_ir::TraitPredicate<RustcInterner<'cx>>,
        >,
        index: usize,
    ) -> Self {
        todo!()
    }
}

impl<'cx> inherent::IntoKind for RustcPredicate<'cx> {
    type Kind =
        rustc_type_ir::Binder<RustcInterner<'cx>, rustc_type_ir::PredicateKind<RustcInterner<'cx>>>;

    fn kind(self) -> Self::Kind {
        todo!()
    }
}

impl<'cx> visit::Flags for RustcPredicate<'cx> {
    fn flags(&self) -> rustc_type_ir::TypeFlags {
        todo!()
    }

    fn outer_exclusive_binder(&self) -> rustc_type_ir::DebruijnIndex {
        todo!()
    }
}

impl<'cx> fold::TypeSuperFoldable<RustcInterner<'cx>> for RustcPredicate<'cx> {
    fn try_super_fold_with<F: fold::FallibleTypeFolder<RustcInterner<'cx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        todo!()
    }
}

impl<'cx> visit::TypeSuperVisitable<RustcInterner<'cx>> for RustcPredicate<'cx> {
    fn super_visit_with<V: visit::TypeVisitor<RustcInterner<'cx>>>(
        &self,
        visitor: &mut V,
    ) -> V::Result {
        todo!()
    }
}

impl<'cx>
    rustc_type_ir::UpcastFrom<RustcInterner<'cx>, rustc_type_ir::PredicateKind<RustcInterner<'cx>>>
    for RustcPredicate<'cx>
{
    fn upcast_from(
        from: rustc_type_ir::PredicateKind<RustcInterner<'_>>,
        interner: RustcInterner<'_>,
    ) -> Self {
        todo!()
    }
}

impl<'cx>
    rustc_type_ir::UpcastFrom<
        RustcInterner<'cx>,
        rustc_type_ir::Binder<RustcInterner<'cx>, rustc_type_ir::PredicateKind<RustcInterner<'cx>>>,
    > for RustcPredicate<'cx>
{
    fn upcast_from(
        from: rustc_type_ir::Binder<
            RustcInterner<'_>,
            rustc_type_ir::PredicateKind<RustcInterner<'_>>,
        >,
        interner: RustcInterner<'_>,
    ) -> Self {
        todo!()
    }
}

impl<'cx>
    rustc_type_ir::UpcastFrom<RustcInterner<'cx>, rustc_type_ir::ClauseKind<RustcInterner<'cx>>>
    for RustcPredicate<'cx>
{
    fn upcast_from(
        from: rustc_type_ir::ClauseKind<RustcInterner<'cx>>,
        interner: RustcInterner<'cx>,
    ) -> Self {
        todo!()
    }
}

impl<'cx>
    rustc_type_ir::UpcastFrom<
        RustcInterner<'cx>,
        rustc_type_ir::Binder<RustcInterner<'cx>, rustc_type_ir::ClauseKind<RustcInterner<'cx>>>,
    > for RustcPredicate<'cx>
{
    fn upcast_from(
        from: rustc_type_ir::Binder<
            RustcInterner<'cx>,
            rustc_type_ir::ClauseKind<RustcInterner<'cx>>,
        >,
        interner: RustcInterner<'cx>,
    ) -> Self {
        todo!()
    }
}

impl<'cx> rustc_type_ir::UpcastFrom<RustcInterner<'cx>, RustcClause<'cx>> for RustcPredicate<'cx> {
    fn upcast_from(from: RustcClause<'cx>, interner: RustcInterner<'cx>) -> Self {
        todo!()
    }
}

impl<'cx>
    rustc_type_ir::UpcastFrom<RustcInterner<'cx>, rustc_type_ir::NormalizesTo<RustcInterner<'cx>>>
    for RustcPredicate<'cx>
{
    fn upcast_from(
        from: rustc_type_ir::NormalizesTo<RustcInterner<'cx>>,
        interner: RustcInterner<'cx>,
    ) -> Self {
        todo!()
    }
}

impl<'cx> rustc_type_ir::UpcastFrom<RustcInterner<'cx>, rustc_type_ir::TraitRef<RustcInterner<'cx>>>
    for RustcPredicate<'cx>
{
    fn upcast_from(
        from: rustc_type_ir::TraitRef<RustcInterner<'cx>>,
        interner: RustcInterner<'cx>,
    ) -> Self {
        todo!()
    }
}

impl<'cx>
    rustc_type_ir::UpcastFrom<
        RustcInterner<'cx>,
        rustc_type_ir::Binder<RustcInterner<'cx>, rustc_type_ir::TraitRef<RustcInterner<'cx>>>,
    > for RustcPredicate<'cx>
{
    fn upcast_from(
        from: rustc_type_ir::Binder<
            RustcInterner<'cx>,
            rustc_type_ir::TraitRef<RustcInterner<'cx>>,
        >,
        interner: RustcInterner<'cx>,
    ) -> Self {
        todo!()
    }
}

impl<'cx>
    rustc_type_ir::UpcastFrom<RustcInterner<'cx>, rustc_type_ir::TraitPredicate<RustcInterner<'cx>>>
    for RustcPredicate<'cx>
{
    fn upcast_from(
        from: rustc_type_ir::TraitPredicate<RustcInterner<'cx>>,
        interner: RustcInterner<'cx>,
    ) -> Self {
        todo!()
    }
}

impl<'cx>
    rustc_type_ir::UpcastFrom<
        RustcInterner<'cx>,
        rustc_type_ir::OutlivesPredicate<RustcInterner<'cx>, RustcTy<'cx>>,
    > for RustcPredicate<'cx>
{
    fn upcast_from(
        from: rustc_type_ir::OutlivesPredicate<RustcInterner<'cx>, RustcTy<'cx>>,
        interner: RustcInterner<'cx>,
    ) -> Self {
        todo!()
    }
}

impl<'cx>
    rustc_type_ir::UpcastFrom<
        RustcInterner<'cx>,
        rustc_type_ir::OutlivesPredicate<RustcInterner<'cx>, RustcRegion<'cx>>,
    > for RustcPredicate<'cx>
{
    fn upcast_from(
        from: rustc_type_ir::OutlivesPredicate<RustcInterner<'cx>, RustcRegion<'cx>>,
        interner: RustcInterner<'cx>,
    ) -> Self {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RustcClause<'cx>(&'cx ());

todo_structural!(RustcClause<'cx>);

impl<'cx> inherent::Clause<RustcInterner<'cx>> for RustcClause<'cx> {
    fn as_predicate(self) -> <RustcInterner<'cx> as rustc_type_ir::Interner>::Predicate {
        todo!()
    }

    fn instantiate_supertrait(
        self,
        cx: RustcInterner<'cx>,
        trait_ref: rustc_type_ir::Binder<
            RustcInterner<'cx>,
            rustc_type_ir::TraitRef<RustcInterner<'cx>>,
        >,
    ) -> Self {
        todo!()
    }
}

impl<'cx> elaborate::Elaboratable<RustcInterner<'cx>> for RustcClause<'cx> {
    fn predicate_kind(
        self,
    ) -> rustc_type_ir::Binder<RustcInterner<'cx>, rustc_type_ir::PredicateKind<RustcInterner<'cx>>>
    {
        todo!()
    }

    fn as_clause(self) -> Option<<RustcInterner<'cx> as rustc_type_ir::Interner>::Clause> {
        todo!()
    }

    fn child(&self, clause: <RustcInterner<'cx> as rustc_type_ir::Interner>::Clause) -> Self {
        todo!()
    }

    fn child_with_derived_cause(
        &self,
        clause: <RustcInterner<'_> as rustc_type_ir::Interner>::Clause,
        span: <RustcInterner<'_> as rustc_type_ir::Interner>::Span,
        parent_trait_pred: rustc_type_ir::Binder<
            RustcInterner<'_>,
            rustc_type_ir::TraitPredicate<RustcInterner<'_>>,
        >,
        index: usize,
    ) -> Self {
        todo!()
    }
}

impl<'cx> inherent::IntoKind for RustcClause<'cx> {
    type Kind =
        rustc_type_ir::Binder<RustcInterner<'cx>, rustc_type_ir::ClauseKind<RustcInterner<'cx>>>;

    fn kind(self) -> Self::Kind {
        todo!()
    }
}

impl<'cx> visit::Flags for RustcClause<'cx> {
    fn flags(&self) -> rustc_type_ir::TypeFlags {
        todo!()
    }

    fn outer_exclusive_binder(&self) -> rustc_type_ir::DebruijnIndex {
        todo!()
    }
}

impl<'cx> fold::TypeSuperFoldable<RustcInterner<'cx>> for RustcClause<'cx> {
    fn try_super_fold_with<F: fold::FallibleTypeFolder<RustcInterner<'cx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        todo!()
    }
}

impl<'cx> visit::TypeSuperVisitable<RustcInterner<'cx>> for RustcClause<'cx> {
    fn super_visit_with<V: visit::TypeVisitor<RustcInterner<'cx>>>(
        &self,
        visitor: &mut V,
    ) -> V::Result {
        todo!()
    }
}

impl<'cx>
    rustc_type_ir::UpcastFrom<
        RustcInterner<'cx>,
        rustc_type_ir::Binder<RustcInterner<'_>, rustc_type_ir::ClauseKind<RustcInterner<'_>>>,
    > for RustcClause<'cx>
{
    fn upcast_from(
        from: rustc_type_ir::Binder<
            RustcInterner<'_>,
            rustc_type_ir::ClauseKind<RustcInterner<'_>>,
        >,
        interner: RustcInterner<'_>,
    ) -> Self {
        todo!()
    }
}

impl<'cx> rustc_type_ir::UpcastFrom<RustcInterner<'_>, rustc_type_ir::TraitRef<RustcInterner<'_>>>
    for RustcClause<'cx>
{
    fn upcast_from(
        from: rustc_type_ir::TraitRef<RustcInterner<'_>>,
        interner: RustcInterner<'_>,
    ) -> Self {
        todo!()
    }
}

impl<'cx>
    rustc_type_ir::UpcastFrom<
        RustcInterner<'_>,
        rustc_type_ir::Binder<RustcInterner<'_>, rustc_type_ir::TraitRef<RustcInterner<'_>>>,
    > for RustcClause<'cx>
{
    fn upcast_from(
        from: rustc_type_ir::Binder<RustcInterner<'_>, rustc_type_ir::TraitRef<RustcInterner<'_>>>,
        interner: RustcInterner<'_>,
    ) -> Self {
        todo!()
    }
}

impl<'cx>
    rustc_type_ir::UpcastFrom<RustcInterner<'_>, rustc_type_ir::TraitPredicate<RustcInterner<'_>>>
    for RustcClause<'cx>
{
    fn upcast_from(
        from: rustc_type_ir::TraitPredicate<RustcInterner<'_>>,
        interner: RustcInterner<'_>,
    ) -> Self {
        todo!()
    }
}

impl<'cx>
    rustc_type_ir::UpcastFrom<
        RustcInterner<'_>,
        rustc_type_ir::Binder<RustcInterner<'_>, rustc_type_ir::TraitPredicate<RustcInterner<'_>>>,
    > for RustcClause<'cx>
{
    fn upcast_from(
        from: rustc_type_ir::Binder<
            RustcInterner<'_>,
            rustc_type_ir::TraitPredicate<RustcInterner<'_>>,
        >,
        interner: RustcInterner<'_>,
    ) -> Self {
        todo!()
    }
}

impl<'cx>
    rustc_type_ir::UpcastFrom<
        RustcInterner<'_>,
        rustc_type_ir::ProjectionPredicate<RustcInterner<'_>>,
    > for RustcClause<'cx>
{
    fn upcast_from(
        from: rustc_type_ir::ProjectionPredicate<RustcInterner<'_>>,
        interner: RustcInterner<'_>,
    ) -> Self {
        todo!()
    }
}

impl<'cx>
    rustc_type_ir::UpcastFrom<
        RustcInterner<'_>,
        rustc_type_ir::Binder<
            RustcInterner<'_>,
            rustc_type_ir::ProjectionPredicate<RustcInterner<'_>>,
        >,
    > for RustcClause<'cx>
{
    fn upcast_from(
        from: rustc_type_ir::Binder<
            RustcInterner<'_>,
            rustc_type_ir::ProjectionPredicate<RustcInterner<'_>>,
        >,
        interner: RustcInterner<'_>,
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

impl<'cx> fold::TypeSuperFoldable<RustcInterner<'cx>> for RustcClauses {
    fn try_super_fold_with<F: fold::FallibleTypeFolder<RustcInterner<'cx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        todo!()
    }
}

impl<'cx> visit::TypeSuperVisitable<RustcInterner<'cx>> for RustcClauses {
    fn super_visit_with<V: visit::TypeVisitor<RustcInterner<'cx>>>(
        &self,
        visitor: &mut V,
    ) -> V::Result {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RustcGenericsOf;

todo_structural!(RustcGenericsOf);

impl inherent::GenericsOf<RustcInterner<'_>> for RustcGenericsOf {
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

impl<'cx> inherent::AdtDef<RustcInterner<'cx>> for RustcAdtDef {
    fn def_id(&self) -> <RustcInterner<'cx> as rustc_type_ir::Interner>::DefId {
        todo!()
    }

    fn is_struct(&self) -> bool {
        todo!()
    }

    fn struct_tail_ty(
        self,
        interner: RustcInterner<'cx>,
    ) -> Option<
        rustc_type_ir::EarlyBinder<
            RustcInterner<'cx>,
            <RustcInterner<'cx> as rustc_type_ir::Interner>::Ty,
        >,
    > {
        todo!()
    }

    fn is_phantom_data(&self) -> bool {
        todo!()
    }

    fn all_field_tys(
        self,
        interner: RustcInterner<'cx>,
    ) -> rustc_type_ir::EarlyBinder<
        RustcInterner<'cx>,
        impl IntoIterator<Item = <RustcInterner<'cx> as rustc_type_ir::Interner>::Ty>,
    > {
        todo!();
        rustc_type_ir::EarlyBinder::bind(None)
    }

    fn sized_constraint(
        self,
        interner: RustcInterner<'cx>,
    ) -> Option<
        rustc_type_ir::EarlyBinder<
            RustcInterner<'cx>,
            <RustcInterner<'cx> as rustc_type_ir::Interner>::Ty,
        >,
    > {
        todo!()
    }

    fn is_fundamental(&self) -> bool {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RustcFeatures;

impl inherent::Features<RustcInterner<'_>> for RustcFeatures {
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

impl<'cx> rustc_type_ir::Interner for RustcInterner<'cx> {
    type DefId = InternId;
    type LocalDefId = InternId;
    type Span = RustcSpan;

    type GenericArgs = GenericArgs<'cx>;
    type GenericArgsSlice = GenericArgs<'cx>;
    type GenericArg = RustcGenericArg<'cx>;

    type Term = RustcTerm<'cx>;

    type BoundVarKinds = RustcBoundVarKinds;
    type BoundVarKind = RustcBoundVarKind;

    type PredefinedOpaques = RustcPredefinedOpaques<'cx>;

    fn mk_predefined_opaques_in_body(
        self,
        data: rustc_type_ir::solve::PredefinedOpaquesData<Self>,
    ) -> Self::PredefinedOpaques {
        todo!()
    }

    type DefiningOpaqueTypes = RustcDefiningOpaqueTypes;

    type CanonicalVars = RustcCanonicalVars<'cx>;

    fn mk_canonical_var_infos(
        self,
        infos: &[rustc_type_ir::CanonicalVarInfo<Self>],
    ) -> Self::CanonicalVars {
        todo!()
    }

    type ExternalConstraints = RustcExternalConstraints<'cx>;

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

    type Ty = RustcTy<'cx>;
    type Tys = RustcTys<'cx>;
    type FnInputTys = RustcFnInputTys<'cx>;
    type ParamTy = RustcParamTy;
    type BoundTy = RustcBoundTy;
    type PlaceholderTy = RustcPlaceholderTy;

    type ErrorGuaranteed = RustcErrorGuaranteed;
    type BoundExistentialPredicates = RustcBoundExistentialPredicates<'cx>;
    type AllocId = RustcAllocId;
    type Pat = RustcPat;
    type Safety = RustcSafety;
    type Abi = RustcAbi;

    type Const = RustcConst<'cx>;
    type PlaceholderConst = RustcPlaceholderConst;
    type ParamConst = RustcParamConst;
    type BoundConst = RustcBoundConst;
    type ValueConst = RustcValueConst;
    type ExprConst = RustcExprConst;

    type Region = RustcRegion<'cx>;
    type EarlyParamRegion = RustcEarlyParamRegion;
    type LateParamRegion = RustcLateParamRegion;
    type BoundRegion = RustcBoundRegion;
    type PlaceholderRegion = RustcPlaceholderRegion;

    type ParamEnv = RustcParamEnv;
    type Predicate = RustcPredicate<'cx>;
    type Clause = RustcClause<'cx>;
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
