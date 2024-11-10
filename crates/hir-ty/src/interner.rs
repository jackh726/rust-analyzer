#![allow(unused)]
//! Implementation of the Chalk `Interner` trait, which allows customizing the
//! representation of the various objects Chalk deals with (types, goals etc.).

use crate::{
    chalk_db, tls, AliasTy, CanonicalVarKind, CanonicalVarKinds, ClosureId, Const, ConstData,
    ConstScalar, FnAbi, FnDefId, GenericArg, GenericArgData, Goal, GoalData, InEnvironment,
    Lifetime, LifetimeData, OpaqueTy, OpaqueTyId, ProgramClause, ProjectionTy,
    QuantifiedWhereClause, QuantifiedWhereClauses, Substitution, Ty, TyKind, VariableKind,
};
use base_db::ra_salsa::InternId;
use chalk_ir::{ProgramClauseImplication, SeparatorTraitRef, Variance};
use hir_def::TypeAliasId;
use intern::{impl_internable, Interned};
use rustc_ast_ir::visit::VisitorResult;
use rustc_index_in_tree::bit_set::BitSet;
use rustc_type_ir::{
    elaborate, fold, inherent, ir_print, relate,
    solve::{ExternalConstraintsData, PredefinedOpaquesData},
    visit, CanonicalVarInfo, ConstKind, GenericArgKind, TermKind,
};
use smallvec::SmallVec;
use span::Span;
use std::fmt;
use triomphe::Arc;

type TyData = chalk_ir::TyData<Interner>;
type VariableKinds = chalk_ir::VariableKinds<Interner>;
type Goals = chalk_ir::Goals<Interner>;
type ProgramClauseData = chalk_ir::ProgramClauseData<Interner>;
type Constraint = chalk_ir::Constraint<Interner>;
type Constraints = chalk_ir::Constraints<Interner>;
type ProgramClauses = chalk_ir::ProgramClauses<Interner>;

#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct Interner;

#[derive(PartialEq, Eq, Hash)]
pub struct InternedWrapper<T>(T);

impl<T: fmt::Debug> fmt::Debug for InternedWrapper<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

impl<T> std::ops::Deref for InternedWrapper<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl_internable!(
    InternedWrapper<Vec<VariableKind>>,
    InternedWrapper<SmallVec<[GenericArg; 2]>>,
    InternedWrapper<TyData>,
    InternedWrapper<LifetimeData>,
    InternedWrapper<ConstData>,
    InternedWrapper<ConstScalar>,
    InternedWrapper<Vec<CanonicalVarKind>>,
    InternedWrapper<Vec<ProgramClause>>,
    InternedWrapper<Vec<QuantifiedWhereClause>>,
    InternedWrapper<SmallVec<[Variance; 16]>>,
);

impl chalk_ir::interner::Interner for Interner {
    type InternedType = Interned<InternedWrapper<TyData>>;
    type InternedLifetime = Interned<InternedWrapper<LifetimeData>>;
    type InternedConst = Interned<InternedWrapper<ConstData>>;
    type InternedConcreteConst = ConstScalar;
    type InternedGenericArg = GenericArgData;
    // We could do the following, but that saves "only" 20mb on self while increasing inferecene
    // time by ~2.5%
    // type InternedGoal = Interned<InternedWrapper<GoalData>>;
    type InternedGoal = Arc<GoalData>;
    type InternedGoals = Vec<Goal>;
    type InternedSubstitution = Interned<InternedWrapper<SmallVec<[GenericArg; 2]>>>;
    type InternedProgramClauses = Interned<InternedWrapper<Vec<ProgramClause>>>;
    type InternedProgramClause = ProgramClauseData;
    type InternedQuantifiedWhereClauses = Interned<InternedWrapper<Vec<QuantifiedWhereClause>>>;
    type InternedVariableKinds = Interned<InternedWrapper<Vec<VariableKind>>>;
    type InternedCanonicalVarKinds = Interned<InternedWrapper<Vec<CanonicalVarKind>>>;
    type InternedConstraints = Vec<InEnvironment<Constraint>>;
    type InternedVariances = SmallVec<[Variance; 16]>;
    type DefId = InternId;
    type InternedAdtId = hir_def::AdtId;
    type Identifier = TypeAliasId;
    type FnAbi = FnAbi;

    fn debug_adt_id(
        type_kind_id: chalk_db::AdtId,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        tls::with_current_program(|prog| Some(prog?.debug_struct_id(type_kind_id, fmt)))
    }

    fn debug_trait_id(
        type_kind_id: chalk_db::TraitId,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        tls::with_current_program(|prog| Some(prog?.debug_trait_id(type_kind_id, fmt)))
    }

    fn debug_assoc_type_id(
        id: chalk_db::AssocTypeId,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        tls::with_current_program(|prog| Some(prog?.debug_assoc_type_id(id, fmt)))
    }

    fn debug_opaque_ty_id(
        opaque_ty_id: OpaqueTyId,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "OpaqueTy#{}", opaque_ty_id.0))
    }

    fn debug_fn_def_id(fn_def_id: FnDefId, fmt: &mut fmt::Formatter<'_>) -> Option<fmt::Result> {
        tls::with_current_program(|prog| Some(prog?.debug_fn_def_id(fn_def_id, fmt)))
    }

    fn debug_closure_id(
        _fn_def_id: ClosureId,
        _fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        None
    }

    fn debug_alias(alias: &AliasTy, fmt: &mut fmt::Formatter<'_>) -> Option<fmt::Result> {
        use std::fmt::Debug;
        match alias {
            AliasTy::Projection(projection_ty) => Interner::debug_projection_ty(projection_ty, fmt),
            AliasTy::Opaque(opaque_ty) => Some(opaque_ty.fmt(fmt)),
        }
    }

    fn debug_projection_ty(
        proj: &ProjectionTy,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        tls::with_current_program(|prog| Some(prog?.debug_projection_ty(proj, fmt)))
    }

    fn debug_opaque_ty(opaque_ty: &OpaqueTy, fmt: &mut fmt::Formatter<'_>) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", opaque_ty.opaque_ty_id))
    }

    fn debug_ty(ty: &Ty, fmt: &mut fmt::Formatter<'_>) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", ty.data(Interner)))
    }

    fn debug_lifetime(lifetime: &Lifetime, fmt: &mut fmt::Formatter<'_>) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", lifetime.data(Interner)))
    }

    fn debug_const(constant: &Const, fmt: &mut fmt::Formatter<'_>) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", constant.data(Interner)))
    }

    fn debug_generic_arg(
        parameter: &GenericArg,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", parameter.data(Interner).inner_debug()))
    }

    fn debug_variable_kinds(
        variable_kinds: &VariableKinds,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", variable_kinds.as_slice(Interner)))
    }

    fn debug_variable_kinds_with_angles(
        variable_kinds: &VariableKinds,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", variable_kinds.inner_debug(Interner)))
    }

    fn debug_canonical_var_kinds(
        canonical_var_kinds: &CanonicalVarKinds,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", canonical_var_kinds.as_slice(Interner)))
    }
    fn debug_goal(goal: &Goal, fmt: &mut fmt::Formatter<'_>) -> Option<fmt::Result> {
        let goal_data = goal.data(Interner);
        Some(write!(fmt, "{goal_data:?}"))
    }
    fn debug_goals(goals: &Goals, fmt: &mut fmt::Formatter<'_>) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", goals.debug(Interner)))
    }
    fn debug_program_clause_implication(
        pci: &ProgramClauseImplication<Self>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", pci.debug(Interner)))
    }
    fn debug_program_clause(
        clause: &ProgramClause,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", clause.data(Interner)))
    }
    fn debug_program_clauses(
        clauses: &ProgramClauses,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", clauses.as_slice(Interner)))
    }
    fn debug_substitution(
        substitution: &Substitution,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", substitution.debug(Interner)))
    }
    fn debug_separator_trait_ref(
        separator_trait_ref: &SeparatorTraitRef<'_, Interner>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", separator_trait_ref.debug(Interner)))
    }

    fn debug_quantified_where_clauses(
        clauses: &QuantifiedWhereClauses,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", clauses.as_slice(Interner)))
    }

    fn debug_constraints(
        _clauses: &Constraints,
        _fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        None
    }

    fn intern_ty(self, kind: TyKind) -> Self::InternedType {
        let flags = kind.compute_flags(self);
        Interned::new(InternedWrapper(TyData { kind, flags }))
    }

    fn ty_data(self, ty: &Self::InternedType) -> &TyData {
        &ty.0
    }

    fn intern_lifetime(self, lifetime: LifetimeData) -> Self::InternedLifetime {
        Interned::new(InternedWrapper(lifetime))
    }

    fn lifetime_data(self, lifetime: &Self::InternedLifetime) -> &LifetimeData {
        &lifetime.0
    }

    fn intern_const(self, constant: ConstData) -> Self::InternedConst {
        Interned::new(InternedWrapper(constant))
    }

    fn const_data(self, constant: &Self::InternedConst) -> &ConstData {
        &constant.0
    }

    fn const_eq(
        self,
        _ty: &Self::InternedType,
        c1: &Self::InternedConcreteConst,
        c2: &Self::InternedConcreteConst,
    ) -> bool {
        !matches!(c1, ConstScalar::Bytes(..)) || !matches!(c2, ConstScalar::Bytes(..)) || (c1 == c2)
    }

    fn intern_generic_arg(self, parameter: GenericArgData) -> Self::InternedGenericArg {
        parameter
    }

    fn generic_arg_data(self, parameter: &Self::InternedGenericArg) -> &GenericArgData {
        parameter
    }

    fn intern_goal(self, goal: GoalData) -> Self::InternedGoal {
        Arc::new(goal)
    }

    fn goal_data(self, goal: &Self::InternedGoal) -> &GoalData {
        goal
    }

    fn intern_goals<E>(
        self,
        data: impl IntoIterator<Item = Result<Goal, E>>,
    ) -> Result<Self::InternedGoals, E> {
        // let hash =
        //     std::hash::BuildHasher::hash_one(&BuildHasherDefault::<FxHasher>::default(), &goal);
        // Interned::new(InternedWrapper(PreHashedWrapper(goal, hash)))
        data.into_iter().collect()
    }

    fn goals_data(self, goals: &Self::InternedGoals) -> &[Goal] {
        goals
    }

    fn intern_substitution<E>(
        self,
        data: impl IntoIterator<Item = Result<GenericArg, E>>,
    ) -> Result<Self::InternedSubstitution, E> {
        Ok(Interned::new(InternedWrapper(data.into_iter().collect::<Result<_, _>>()?)))
    }

    fn substitution_data(self, substitution: &Self::InternedSubstitution) -> &[GenericArg] {
        &substitution.as_ref().0
    }

    fn intern_program_clause(self, data: ProgramClauseData) -> Self::InternedProgramClause {
        data
    }

    fn program_clause_data(self, clause: &Self::InternedProgramClause) -> &ProgramClauseData {
        clause
    }

    fn intern_program_clauses<E>(
        self,
        data: impl IntoIterator<Item = Result<ProgramClause, E>>,
    ) -> Result<Self::InternedProgramClauses, E> {
        Ok(Interned::new(InternedWrapper(data.into_iter().collect::<Result<_, _>>()?)))
    }

    fn program_clauses_data(self, clauses: &Self::InternedProgramClauses) -> &[ProgramClause] {
        clauses
    }

    fn intern_quantified_where_clauses<E>(
        self,
        data: impl IntoIterator<Item = Result<QuantifiedWhereClause, E>>,
    ) -> Result<Self::InternedQuantifiedWhereClauses, E> {
        Ok(Interned::new(InternedWrapper(data.into_iter().collect::<Result<_, _>>()?)))
    }

    fn quantified_where_clauses_data(
        self,
        clauses: &Self::InternedQuantifiedWhereClauses,
    ) -> &[QuantifiedWhereClause] {
        clauses
    }

    fn intern_generic_arg_kinds<E>(
        self,
        data: impl IntoIterator<Item = Result<VariableKind, E>>,
    ) -> Result<Self::InternedVariableKinds, E> {
        Ok(Interned::new(InternedWrapper(data.into_iter().collect::<Result<_, _>>()?)))
    }

    fn variable_kinds_data(self, parameter_kinds: &Self::InternedVariableKinds) -> &[VariableKind] {
        &parameter_kinds.as_ref().0
    }

    fn intern_canonical_var_kinds<E>(
        self,
        data: impl IntoIterator<Item = Result<CanonicalVarKind, E>>,
    ) -> Result<Self::InternedCanonicalVarKinds, E> {
        Ok(Interned::new(InternedWrapper(data.into_iter().collect::<Result<_, _>>()?)))
    }

    fn canonical_var_kinds_data(
        self,
        canonical_var_kinds: &Self::InternedCanonicalVarKinds,
    ) -> &[CanonicalVarKind] {
        canonical_var_kinds
    }
    fn intern_constraints<E>(
        self,
        data: impl IntoIterator<Item = Result<InEnvironment<Constraint>, E>>,
    ) -> Result<Self::InternedConstraints, E> {
        data.into_iter().collect()
    }
    fn constraints_data(
        self,
        constraints: &Self::InternedConstraints,
    ) -> &[InEnvironment<Constraint>] {
        constraints
    }

    fn intern_variances<E>(
        self,
        data: impl IntoIterator<Item = Result<Variance, E>>,
    ) -> Result<Self::InternedVariances, E> {
        data.into_iter().collect::<Result<_, _>>()
    }

    fn variances_data(self, variances: &Self::InternedVariances) -> &[Variance] {
        variances
    }
}

macro_rules! todo_structural {
    ($t:ty) => {
        impl relate::Relate<Interner> for $t {
            fn relate<R: relate::TypeRelation<Interner>>(
                _relation: &mut R,
                _a: Self,
                _b: Self,
            ) -> relate::RelateResult<Interner, Self> {
                todo!()
            }
        }

        impl fold::TypeFoldable<Interner> for $t {
            fn try_fold_with<F: fold::FallibleTypeFolder<Interner>>(
                self,
                _folder: &mut F,
            ) -> Result<Self, F::Error> {
                todo!()
            }
        }

        impl visit::TypeVisitable<Interner> for $t {
            fn visit_with<V: visit::TypeVisitor<Interner>>(&self, _visitor: &mut V) -> V::Result {
                todo!()
            }
        }
    };
}

impl inherent::DefId<Interner> for InternId {
    fn as_local(self) -> Option<InternId> {
        Some(self)
    }
    fn is_local(self) -> bool {
        true
    }
}

todo_structural!(InternId);

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct RustcSpan(Span);

todo_structural!(RustcSpan);

impl inherent::Span<Interner> for RustcSpan {
    fn dummy() -> Self {
        todo!()
    }
}

type InternedGenericArgs = Interned<InternedWrapper<SmallVec<[GenericArg; 2]>>>;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct GenericArgs(InternedGenericArgs);

impl inherent::GenericArgs<Interner> for GenericArgs {
    fn dummy() -> Self {
        todo!()
    }

    fn rebase_onto(
        self,
        interner: Interner,
        source_def_id: <Interner as rustc_type_ir::Interner>::DefId,
        target: <Interner as rustc_type_ir::Interner>::GenericArgs,
    ) -> <Interner as rustc_type_ir::Interner>::GenericArgs {
        todo!()
    }

    fn type_at(self, i: usize) -> <Interner as rustc_type_ir::Interner>::Ty {
        todo!()
    }

    fn region_at(self, i: usize) -> <Interner as rustc_type_ir::Interner>::Region {
        todo!()
    }

    fn const_at(self, i: usize) -> <Interner as rustc_type_ir::Interner>::Const {
        todo!()
    }

    fn identity_for_item(
        interner: Interner,
        def_id: <Interner as rustc_type_ir::Interner>::DefId,
    ) -> <Interner as rustc_type_ir::Interner>::GenericArgs {
        todo!()
    }

    fn extend_with_error(
        interner: Interner,
        def_id: <Interner as rustc_type_ir::Interner>::DefId,
        original_args: &[<Interner as rustc_type_ir::Interner>::GenericArg],
    ) -> <Interner as rustc_type_ir::Interner>::GenericArgs {
        todo!()
    }

    fn split_closure_args(self) -> rustc_type_ir::ClosureArgsParts<Interner> {
        todo!()
    }

    fn split_coroutine_closure_args(self) -> rustc_type_ir::CoroutineClosureArgsParts<Interner> {
        todo!()
    }

    fn split_coroutine_args(self) -> rustc_type_ir::CoroutineArgsParts<Interner> {
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
pub struct RustcGenericArg(chalk_ir::GenericArg<Interner>);

todo_structural!(RustcGenericArg);

impl inherent::GenericArg<Interner> for RustcGenericArg {}

impl inherent::IntoKind for RustcGenericArg {
    type Kind = GenericArgKind<Interner>;

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

impl inherent::Ty<Interner> for RustcTy {
    fn new_unit(interner: Interner) -> Self {
        todo!()
    }

    fn new_bool(interner: Interner) -> Self {
        todo!()
    }

    fn new_u8(interner: Interner) -> Self {
        todo!()
    }

    fn new_usize(interner: Interner) -> Self {
        todo!()
    }

    fn new_infer(interner: Interner, var: rustc_type_ir::InferTy) -> Self {
        todo!()
    }

    fn new_var(interner: Interner, var: rustc_type_ir::TyVid) -> Self {
        todo!()
    }

    fn new_param(
        interner: Interner,
        param: <Interner as rustc_type_ir::Interner>::ParamTy,
    ) -> Self {
        todo!()
    }

    fn new_placeholder(
        interner: Interner,
        param: <Interner as rustc_type_ir::Interner>::PlaceholderTy,
    ) -> Self {
        todo!()
    }

    fn new_bound(
        interner: Interner,
        debruijn: rustc_type_ir::DebruijnIndex,
        var: <Interner as rustc_type_ir::Interner>::BoundTy,
    ) -> Self {
        todo!()
    }

    fn new_anon_bound(
        interner: Interner,
        debruijn: rustc_type_ir::DebruijnIndex,
        var: rustc_type_ir::BoundVar,
    ) -> Self {
        todo!()
    }

    fn new_alias(
        interner: Interner,
        kind: rustc_type_ir::AliasTyKind,
        alias_ty: rustc_type_ir::AliasTy<Interner>,
    ) -> Self {
        todo!()
    }

    fn new_error(
        interner: Interner,
        guar: <Interner as rustc_type_ir::Interner>::ErrorGuaranteed,
    ) -> Self {
        todo!()
    }

    fn new_adt(
        interner: Interner,
        adt_def: <Interner as rustc_type_ir::Interner>::AdtDef,
        args: <Interner as rustc_type_ir::Interner>::GenericArgs,
    ) -> Self {
        todo!()
    }

    fn new_foreign(
        interner: Interner,
        def_id: <Interner as rustc_type_ir::Interner>::DefId,
    ) -> Self {
        todo!()
    }

    fn new_dynamic(
        interner: Interner,
        preds: <Interner as rustc_type_ir::Interner>::BoundExistentialPredicates,
        region: <Interner as rustc_type_ir::Interner>::Region,
        kind: rustc_type_ir::DynKind,
    ) -> Self {
        todo!()
    }

    fn new_coroutine(
        interner: Interner,
        def_id: <Interner as rustc_type_ir::Interner>::DefId,
        args: <Interner as rustc_type_ir::Interner>::GenericArgs,
    ) -> Self {
        todo!()
    }

    fn new_coroutine_closure(
        interner: Interner,
        def_id: <Interner as rustc_type_ir::Interner>::DefId,
        args: <Interner as rustc_type_ir::Interner>::GenericArgs,
    ) -> Self {
        todo!()
    }

    fn new_closure(
        interner: Interner,
        def_id: <Interner as rustc_type_ir::Interner>::DefId,
        args: <Interner as rustc_type_ir::Interner>::GenericArgs,
    ) -> Self {
        todo!()
    }

    fn new_coroutine_witness(
        interner: Interner,
        def_id: <Interner as rustc_type_ir::Interner>::DefId,
        args: <Interner as rustc_type_ir::Interner>::GenericArgs,
    ) -> Self {
        todo!()
    }

    fn new_ptr(interner: Interner, ty: Self, mutbl: rustc_ast_ir::Mutability) -> Self {
        todo!()
    }

    fn new_ref(
        interner: Interner,
        region: <Interner as rustc_type_ir::Interner>::Region,
        ty: Self,
        mutbl: rustc_ast_ir::Mutability,
    ) -> Self {
        todo!()
    }

    fn new_array_with_const_len(
        interner: Interner,
        ty: Self,
        len: <Interner as rustc_type_ir::Interner>::Const,
    ) -> Self {
        todo!()
    }

    fn new_slice(interner: Interner, ty: Self) -> Self {
        todo!()
    }

    fn new_tup(interner: Interner, tys: &[<Interner as rustc_type_ir::Interner>::Ty]) -> Self {
        todo!()
    }

    fn new_tup_from_iter<It, T>(interner: Interner, iter: It) -> T::Output
    where
        It: Iterator<Item = T>,
        T: rustc_type_ir::CollectAndApply<Self, Self>,
    {
        todo!()
    }

    fn new_fn_def(
        interner: Interner,
        def_id: <Interner as rustc_type_ir::Interner>::DefId,
        args: <Interner as rustc_type_ir::Interner>::GenericArgs,
    ) -> Self {
        todo!()
    }

    fn new_fn_ptr(
        interner: Interner,
        sig: rustc_type_ir::Binder<Interner, rustc_type_ir::FnSig<Interner>>,
    ) -> Self {
        todo!()
    }

    fn new_pat(
        interner: Interner,
        ty: Self,
        pat: <Interner as rustc_type_ir::Interner>::Pat,
    ) -> Self {
        todo!()
    }

    fn tuple_fields(self) -> <Interner as rustc_type_ir::Interner>::Tys {
        todo!()
    }

    fn to_opt_closure_kind(self) -> Option<rustc_type_ir::ClosureKind> {
        todo!()
    }

    fn from_closure_kind(interner: Interner, kind: rustc_type_ir::ClosureKind) -> Self {
        todo!()
    }

    fn from_coroutine_closure_kind(interner: Interner, kind: rustc_type_ir::ClosureKind) -> Self {
        todo!()
    }

    fn discriminant_ty(self, interner: Interner) -> <Interner as rustc_type_ir::Interner>::Ty {
        todo!()
    }

    fn async_destructor_ty(self, interner: Interner) -> <Interner as rustc_type_ir::Interner>::Ty {
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

impl fold::TypeSuperFoldable<Interner> for RustcTy {
    fn try_super_fold_with<F: fold::FallibleTypeFolder<Interner>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        todo!()
    }
}

impl visit::TypeSuperVisitable<Interner> for RustcTy {
    fn super_visit_with<V: visit::TypeVisitor<Interner>>(&self, visitor: &mut V) -> V::Result {
        todo!()
    }
}

impl inherent::IntoKind for RustcTy {
    type Kind = rustc_type_ir::TyKind<Interner>;

    fn kind(self) -> Self::Kind {
        todo!()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RustcConst;

todo_structural!(RustcConst);

impl inherent::Const<Interner> for RustcConst {
    fn try_to_target_usize(self, interner: Interner) -> Option<u64> {
        todo!()
    }

    fn new_infer(interner: Interner, var: rustc_type_ir::InferConst) -> Self {
        todo!()
    }

    fn new_var(interner: Interner, var: rustc_type_ir::ConstVid) -> Self {
        todo!()
    }

    fn new_bound(
        interner: Interner,
        debruijn: rustc_type_ir::DebruijnIndex,
        var: <Interner as rustc_type_ir::Interner>::BoundConst,
    ) -> Self {
        todo!()
    }

    fn new_anon_bound(
        interner: Interner,
        debruijn: rustc_type_ir::DebruijnIndex,
        var: rustc_type_ir::BoundVar,
    ) -> Self {
        todo!()
    }

    fn new_unevaluated(interner: Interner, uv: rustc_type_ir::UnevaluatedConst<Interner>) -> Self {
        todo!()
    }

    fn new_expr(
        interner: Interner,
        expr: <Interner as rustc_type_ir::Interner>::ExprConst,
    ) -> Self {
        todo!()
    }

    fn new_error(
        interner: Interner,
        guar: <Interner as rustc_type_ir::Interner>::ErrorGuaranteed,
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

impl fold::TypeSuperFoldable<Interner> for RustcConst {
    fn try_super_fold_with<F: fold::FallibleTypeFolder<Interner>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        todo!()
    }
}

impl visit::TypeSuperVisitable<Interner> for RustcConst {
    fn super_visit_with<V: visit::TypeVisitor<Interner>>(&self, visitor: &mut V) -> V::Result {
        todo!()
    }
}

impl inherent::IntoKind for RustcConst {
    type Kind = ConstKind<Interner>;

    fn kind(self) -> Self::Kind {
        todo!()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RustcTerm {
    Ty(RustcTy),
    Const(RustcConst),
}

impl inherent::Term<Interner> for RustcTerm {}

todo_structural!(RustcTerm);

impl inherent::IntoKind for RustcTerm {
    type Kind = TermKind<Interner>;

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

impl<T> ir_print::IrPrint<T> for Interner {
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
    type Target = PredefinedOpaquesData<Interner>;

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
    type Item = CanonicalVarInfo<Interner>;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl inherent::SliceLike for RustcCanonicalVars {
    type Item = CanonicalVarInfo<Interner>;
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
    type Target = ExternalConstraintsData<Interner>;

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

impl inherent::Tys<Interner> for RustcTys {
    fn inputs(self) -> <Interner as rustc_type_ir::Interner>::FnInputTys {
        todo!()
    }

    fn output(self) -> <Interner as rustc_type_ir::Interner>::Ty {
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

impl inherent::BoundVarLike<Interner> for RustcBoundTy {
    fn var(self) -> rustc_type_ir::BoundVar {
        todo!()
    }

    fn assert_eq(self, var: <Interner as rustc_type_ir::Interner>::BoundVarKind) {
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
    type Item = rustc_type_ir::Binder<Interner, rustc_type_ir::ExistentialPredicate<Interner>>;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl inherent::SliceLike for RustcBoundExistentialPredicates {
    type Item = rustc_type_ir::Binder<Interner, rustc_type_ir::ExistentialPredicate<Interner>>;
    type IntoIter = RustcBoundExistentialPredicatesIter;

    fn iter(self) -> Self::IntoIter {
        todo!()
    }

    fn as_slice(&self) -> &[Self::Item] {
        todo!()
    }
}

impl inherent::BoundExistentialPredicates<Interner> for RustcBoundExistentialPredicates {
    fn principal_def_id(&self) -> Option<<Interner as rustc_type_ir::Interner>::DefId> {
        todo!()
    }

    fn principal(
        self,
    ) -> Option<rustc_type_ir::Binder<Interner, rustc_type_ir::ExistentialTraitRef<Interner>>> {
        todo!()
    }

    fn auto_traits(self) -> impl IntoIterator<Item = <Interner as rustc_type_ir::Interner>::DefId> {
        todo!();
        None
    }

    fn projection_bounds(
        self,
    ) -> impl IntoIterator<
        Item = rustc_type_ir::Binder<Interner, rustc_type_ir::ExistentialProjection<Interner>>,
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

impl inherent::Safety<Interner> for RustcSafety {
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

impl inherent::Abi<Interner> for RustcAbi {
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

impl inherent::BoundVarLike<Interner> for RustcBoundConst {
    fn var(self) -> rustc_type_ir::BoundVar {
        todo!()
    }

    fn assert_eq(self, var: <Interner as rustc_type_ir::Interner>::BoundVarKind) {
        todo!()
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct RustcValueConst;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct RustcExprConst;

todo_structural!(RustcExprConst);

impl inherent::ExprConst<Interner> for RustcExprConst {
    fn args(self) -> <Interner as rustc_type_ir::Interner>::GenericArgs {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RustcRegion;

todo_structural!(RustcRegion);

impl inherent::Region<Interner> for RustcRegion {
    fn new_bound(
        interner: Interner,
        debruijn: rustc_type_ir::DebruijnIndex,
        var: <Interner as rustc_type_ir::Interner>::BoundRegion,
    ) -> Self {
        todo!()
    }

    fn new_anon_bound(
        interner: Interner,
        debruijn: rustc_type_ir::DebruijnIndex,
        var: rustc_type_ir::BoundVar,
    ) -> Self {
        todo!()
    }

    fn new_static(interner: Interner) -> Self {
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

impl fold::TypeSuperFoldable<Interner> for RustcRegion {
    fn try_super_fold_with<F: fold::FallibleTypeFolder<Interner>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        todo!()
    }
}

impl visit::TypeSuperVisitable<Interner> for RustcRegion {
    fn super_visit_with<V: visit::TypeVisitor<Interner>>(&self, visitor: &mut V) -> V::Result {
        todo!()
    }
}

impl inherent::IntoKind for RustcRegion {
    type Kind = rustc_type_ir::RegionKind<Interner>;

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

impl inherent::BoundVarLike<Interner> for RustcBoundRegion {
    fn var(self) -> rustc_type_ir::BoundVar {
        todo!()
    }

    fn assert_eq(self, var: <Interner as rustc_type_ir::Interner>::BoundVarKind) {
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

impl inherent::ParamEnv<Interner> for RustcParamEnv {
    fn reveal(&self) -> rustc_type_ir::solve::Reveal {
        todo!()
    }

    fn caller_bounds(
        self,
    ) -> impl IntoIterator<Item = <Interner as rustc_type_ir::Interner>::Clause> {
        todo!();
        None
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RustcPredicate;

todo_structural!(RustcPredicate);

impl inherent::Predicate<Interner> for RustcPredicate {
    fn as_clause(self) -> Option<<Interner as rustc_type_ir::Interner>::Clause> {
        todo!()
    }

    fn is_coinductive(&self, interner: Interner) -> bool {
        todo!()
    }

    fn allow_normalization(&self) -> bool {
        todo!()
    }
}

impl elaborate::Elaboratable<Interner> for RustcPredicate {
    fn predicate_kind(
        self,
    ) -> rustc_type_ir::Binder<Interner, rustc_type_ir::PredicateKind<Interner>> {
        todo!()
    }

    fn as_clause(self) -> Option<<Interner as rustc_type_ir::Interner>::Clause> {
        todo!()
    }

    fn child(&self, clause: <Interner as rustc_type_ir::Interner>::Clause) -> Self {
        todo!()
    }

    fn child_with_derived_cause(
        &self,
        clause: <Interner as rustc_type_ir::Interner>::Clause,
        span: <Interner as rustc_type_ir::Interner>::Span,
        parent_trait_pred: rustc_type_ir::Binder<Interner, rustc_type_ir::TraitPredicate<Interner>>,
        index: usize,
    ) -> Self {
        todo!()
    }
}

impl inherent::IntoKind for RustcPredicate {
    type Kind = rustc_type_ir::Binder<Interner, rustc_type_ir::PredicateKind<Interner>>;

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

impl fold::TypeSuperFoldable<Interner> for RustcPredicate {
    fn try_super_fold_with<F: fold::FallibleTypeFolder<Interner>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        todo!()
    }
}

impl visit::TypeSuperVisitable<Interner> for RustcPredicate {
    fn super_visit_with<V: visit::TypeVisitor<Interner>>(&self, visitor: &mut V) -> V::Result {
        todo!()
    }
}

impl rustc_type_ir::UpcastFrom<Interner, rustc_type_ir::PredicateKind<Interner>>
    for RustcPredicate
{
    fn upcast_from(from: rustc_type_ir::PredicateKind<Interner>, interner: Interner) -> Self {
        todo!()
    }
}

impl
    rustc_type_ir::UpcastFrom<
        Interner,
        rustc_type_ir::Binder<Interner, rustc_type_ir::PredicateKind<Interner>>,
    > for RustcPredicate
{
    fn upcast_from(
        from: rustc_type_ir::Binder<Interner, rustc_type_ir::PredicateKind<Interner>>,
        interner: Interner,
    ) -> Self {
        todo!()
    }
}

impl rustc_type_ir::UpcastFrom<Interner, rustc_type_ir::ClauseKind<Interner>> for RustcPredicate {
    fn upcast_from(from: rustc_type_ir::ClauseKind<Interner>, interner: Interner) -> Self {
        todo!()
    }
}

impl
    rustc_type_ir::UpcastFrom<
        Interner,
        rustc_type_ir::Binder<Interner, rustc_type_ir::ClauseKind<Interner>>,
    > for RustcPredicate
{
    fn upcast_from(
        from: rustc_type_ir::Binder<Interner, rustc_type_ir::ClauseKind<Interner>>,
        interner: Interner,
    ) -> Self {
        todo!()
    }
}

impl rustc_type_ir::UpcastFrom<Interner, RustcClause> for RustcPredicate {
    fn upcast_from(from: RustcClause, interner: Interner) -> Self {
        todo!()
    }
}

impl rustc_type_ir::UpcastFrom<Interner, rustc_type_ir::NormalizesTo<Interner>> for RustcPredicate {
    fn upcast_from(from: rustc_type_ir::NormalizesTo<Interner>, interner: Interner) -> Self {
        todo!()
    }
}

impl rustc_type_ir::UpcastFrom<Interner, rustc_type_ir::TraitRef<Interner>> for RustcPredicate {
    fn upcast_from(from: rustc_type_ir::TraitRef<Interner>, interner: Interner) -> Self {
        todo!()
    }
}

impl
    rustc_type_ir::UpcastFrom<
        Interner,
        rustc_type_ir::Binder<Interner, rustc_type_ir::TraitRef<Interner>>,
    > for RustcPredicate
{
    fn upcast_from(
        from: rustc_type_ir::Binder<Interner, rustc_type_ir::TraitRef<Interner>>,
        interner: Interner,
    ) -> Self {
        todo!()
    }
}

impl rustc_type_ir::UpcastFrom<Interner, rustc_type_ir::TraitPredicate<Interner>>
    for RustcPredicate
{
    fn upcast_from(from: rustc_type_ir::TraitPredicate<Interner>, interner: Interner) -> Self {
        todo!()
    }
}

impl rustc_type_ir::UpcastFrom<Interner, rustc_type_ir::OutlivesPredicate<Interner, RustcTy>>
    for RustcPredicate
{
    fn upcast_from(
        from: rustc_type_ir::OutlivesPredicate<Interner, RustcTy>,
        interner: Interner,
    ) -> Self {
        todo!()
    }
}

impl rustc_type_ir::UpcastFrom<Interner, rustc_type_ir::OutlivesPredicate<Interner, RustcRegion>>
    for RustcPredicate
{
    fn upcast_from(
        from: rustc_type_ir::OutlivesPredicate<Interner, RustcRegion>,
        interner: Interner,
    ) -> Self {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RustcClause;

todo_structural!(RustcClause);

impl inherent::Clause<Interner> for RustcClause {
    fn as_predicate(self) -> <Interner as rustc_type_ir::Interner>::Predicate {
        todo!()
    }

    fn instantiate_supertrait(
        self,
        cx: Interner,
        trait_ref: rustc_type_ir::Binder<Interner, rustc_type_ir::TraitRef<Interner>>,
    ) -> Self {
        todo!()
    }
}

impl elaborate::Elaboratable<Interner> for RustcClause {
    fn predicate_kind(
        self,
    ) -> rustc_type_ir::Binder<Interner, rustc_type_ir::PredicateKind<Interner>> {
        todo!()
    }

    fn as_clause(self) -> Option<<Interner as rustc_type_ir::Interner>::Clause> {
        todo!()
    }

    fn child(&self, clause: <Interner as rustc_type_ir::Interner>::Clause) -> Self {
        todo!()
    }

    fn child_with_derived_cause(
        &self,
        clause: <Interner as rustc_type_ir::Interner>::Clause,
        span: <Interner as rustc_type_ir::Interner>::Span,
        parent_trait_pred: rustc_type_ir::Binder<Interner, rustc_type_ir::TraitPredicate<Interner>>,
        index: usize,
    ) -> Self {
        todo!()
    }
}

impl inherent::IntoKind for RustcClause {
    type Kind = rustc_type_ir::Binder<Interner, rustc_type_ir::ClauseKind<Interner>>;

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

impl fold::TypeSuperFoldable<Interner> for RustcClause {
    fn try_super_fold_with<F: fold::FallibleTypeFolder<Interner>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        todo!()
    }
}

impl visit::TypeSuperVisitable<Interner> for RustcClause {
    fn super_visit_with<V: visit::TypeVisitor<Interner>>(&self, visitor: &mut V) -> V::Result {
        todo!()
    }
}

impl
    rustc_type_ir::UpcastFrom<
        Interner,
        rustc_type_ir::Binder<Interner, rustc_type_ir::ClauseKind<Interner>>,
    > for RustcClause
{
    fn upcast_from(
        from: rustc_type_ir::Binder<Interner, rustc_type_ir::ClauseKind<Interner>>,
        interner: Interner,
    ) -> Self {
        todo!()
    }
}

impl rustc_type_ir::UpcastFrom<Interner, rustc_type_ir::TraitRef<Interner>> for RustcClause {
    fn upcast_from(from: rustc_type_ir::TraitRef<Interner>, interner: Interner) -> Self {
        todo!()
    }
}

impl
    rustc_type_ir::UpcastFrom<
        Interner,
        rustc_type_ir::Binder<Interner, rustc_type_ir::TraitRef<Interner>>,
    > for RustcClause
{
    fn upcast_from(
        from: rustc_type_ir::Binder<Interner, rustc_type_ir::TraitRef<Interner>>,
        interner: Interner,
    ) -> Self {
        todo!()
    }
}

impl rustc_type_ir::UpcastFrom<Interner, rustc_type_ir::TraitPredicate<Interner>> for RustcClause {
    fn upcast_from(from: rustc_type_ir::TraitPredicate<Interner>, interner: Interner) -> Self {
        todo!()
    }
}

impl
    rustc_type_ir::UpcastFrom<
        Interner,
        rustc_type_ir::Binder<Interner, rustc_type_ir::TraitPredicate<Interner>>,
    > for RustcClause
{
    fn upcast_from(
        from: rustc_type_ir::Binder<Interner, rustc_type_ir::TraitPredicate<Interner>>,
        interner: Interner,
    ) -> Self {
        todo!()
    }
}

impl rustc_type_ir::UpcastFrom<Interner, rustc_type_ir::ProjectionPredicate<Interner>>
    for RustcClause
{
    fn upcast_from(from: rustc_type_ir::ProjectionPredicate<Interner>, interner: Interner) -> Self {
        todo!()
    }
}

impl
    rustc_type_ir::UpcastFrom<
        Interner,
        rustc_type_ir::Binder<Interner, rustc_type_ir::ProjectionPredicate<Interner>>,
    > for RustcClause
{
    fn upcast_from(
        from: rustc_type_ir::Binder<Interner, rustc_type_ir::ProjectionPredicate<Interner>>,
        interner: Interner,
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

impl fold::TypeSuperFoldable<Interner> for RustcClauses {
    fn try_super_fold_with<F: fold::FallibleTypeFolder<Interner>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        todo!()
    }
}

impl visit::TypeSuperVisitable<Interner> for RustcClauses {
    fn super_visit_with<V: visit::TypeVisitor<Interner>>(&self, visitor: &mut V) -> V::Result {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RustcGenericsOf;

todo_structural!(RustcGenericsOf);

impl inherent::GenericsOf<Interner> for RustcGenericsOf {
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

impl inherent::AdtDef<Interner> for RustcAdtDef {
    fn def_id(&self) -> <Interner as rustc_type_ir::Interner>::DefId {
        todo!()
    }

    fn is_struct(&self) -> bool {
        todo!()
    }

    fn struct_tail_ty(
        self,
        interner: Interner,
    ) -> Option<rustc_type_ir::EarlyBinder<Interner, <Interner as rustc_type_ir::Interner>::Ty>>
    {
        todo!()
    }

    fn is_phantom_data(&self) -> bool {
        todo!()
    }

    fn all_field_tys(
        self,
        interner: Interner,
    ) -> rustc_type_ir::EarlyBinder<
        Interner,
        impl IntoIterator<Item = <Interner as rustc_type_ir::Interner>::Ty>,
    > {
        todo!();
        rustc_type_ir::EarlyBinder::bind(None)
    }

    fn sized_constraint(
        self,
        interner: Interner,
    ) -> Option<rustc_type_ir::EarlyBinder<Interner, <Interner as rustc_type_ir::Interner>::Ty>>
    {
        todo!()
    }

    fn is_fundamental(&self) -> bool {
        todo!()
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RustcFeatures;

impl inherent::Features<Interner> for RustcFeatures {
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

impl rustc_type_ir::Interner for Interner {
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

impl chalk_ir::interner::HasInterner for Interner {
    type Interner = Self;
}

#[macro_export]
macro_rules! has_interner {
    ($t:ty) => {
        impl HasInterner for $t {
            type Interner = $crate::Interner;
        }
    };
}
