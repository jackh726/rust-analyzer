use rustc_next_trait_solver::delegate::SolverDelegate;
use rustc_type_ir::{
    solve::{Certainty, NoSolution},
    UniverseIndex,
};

use crate::db::HirDatabase;

use super::{
    infer::InferenceTable, Canonical, CanonicalVarInfo, CanonicalVarValues, Const, DbInterner,
    DbIr, GenericArg, GenericArgs, ParamEnv, Predicate, Span, Ty, UnevaluatedConst,
};

pub type Goal<P> = rustc_type_ir::solve::Goal<DbInterner, P>;

#[derive(Clone)]
pub(crate) struct SolverContext<'db> {
    pub(crate) table: InferenceTable<'db>,
}

impl<'db> std::ops::Deref for SolverContext<'db> {
    type Target = InferenceTable<'db>;

    fn deref(&self) -> &Self::Target {
        &self.table
    }
}

impl<'db> SolverDelegate for SolverContext<'db> {
    type Interner = DbInterner;
    type Span = Span;
    type Infcx = InferenceTable<'db>;
    type Ir = DbIr<'db>;

    fn build_with_canonical<V>(
        cx: Self::Ir,
        canonical: &rustc_type_ir::CanonicalQueryInput<Self::Interner, V>,
    ) -> (Self, V, rustc_type_ir::CanonicalVarValues<Self::Interner>)
    where
        V: rustc_type_ir::fold::TypeFoldable<Self::Interner>,
    {
        todo!()
    }

    fn fresh_var_for_kind_with_span(
        &self,
        arg: <Self::Interner as rustc_type_ir::Interner>::GenericArg,
        span: Self::Span,
    ) -> <Self::Interner as rustc_type_ir::Interner>::GenericArg {
        todo!()
    }

    fn leak_check(
        &self,
        max_input_universe: rustc_type_ir::UniverseIndex,
    ) -> Result<(), NoSolution> {
        todo!()
    }

    fn well_formed_goals(
        &self,
        param_env: <Self::Interner as rustc_type_ir::Interner>::ParamEnv,
        arg: <Self::Interner as rustc_type_ir::Interner>::GenericArg,
    ) -> Option<
        Vec<
            rustc_type_ir::solve::Goal<
                Self::Interner,
                <Self::Interner as rustc_type_ir::Interner>::Predicate,
            >,
        >,
    > {
        todo!()
    }

    fn clone_opaque_types_for_query_response(
        &self,
    ) -> Vec<(
        rustc_type_ir::OpaqueTypeKey<Self::Interner>,
        <Self::Interner as rustc_type_ir::Interner>::Ty,
    )> {
        todo!()
    }

    fn make_deduplicated_outlives_constraints(
        &self,
    ) -> Vec<
        rustc_type_ir::OutlivesPredicate<
            Self::Interner,
            <Self::Interner as rustc_type_ir::Interner>::GenericArg,
        >,
    > {
        todo!()
    }

    fn instantiate_canonical<V>(
        &self,
        canonical: rustc_type_ir::Canonical<Self::Interner, V>,
        values: rustc_type_ir::CanonicalVarValues<Self::Interner>,
    ) -> V
    where
        V: rustc_type_ir::fold::TypeFoldable<Self::Interner>,
    {
        todo!()
    }

    fn instantiate_canonical_var_with_infer(
        &self,
        cv_info: rustc_type_ir::CanonicalVarInfo<Self::Interner>,
        universe_map: impl Fn(rustc_type_ir::UniverseIndex) -> rustc_type_ir::UniverseIndex,
    ) -> <Self::Interner as rustc_type_ir::Interner>::GenericArg {
        todo!()
    }

    fn insert_hidden_type(
        &self,
        opaque_type_key: rustc_type_ir::OpaqueTypeKey<Self::Interner>,
        param_env: <Self::Interner as rustc_type_ir::Interner>::ParamEnv,
        hidden_ty: <Self::Interner as rustc_type_ir::Interner>::Ty,
        goals: &mut Vec<
            rustc_type_ir::solve::Goal<
                Self::Interner,
                <Self::Interner as rustc_type_ir::Interner>::Predicate,
            >,
        >,
    ) -> Result<(), NoSolution> {
        todo!()
    }

    fn add_item_bounds_for_hidden_type(
        &self,
        def_id: <Self::Interner as rustc_type_ir::Interner>::DefId,
        args: <Self::Interner as rustc_type_ir::Interner>::GenericArgs,
        param_env: <Self::Interner as rustc_type_ir::Interner>::ParamEnv,
        hidden_ty: <Self::Interner as rustc_type_ir::Interner>::Ty,
        goals: &mut Vec<
            rustc_type_ir::solve::Goal<
                Self::Interner,
                <Self::Interner as rustc_type_ir::Interner>::Predicate,
            >,
        >,
    ) {
        todo!()
    }

    fn inject_new_hidden_type_unchecked(
        &self,
        key: rustc_type_ir::OpaqueTypeKey<Self::Interner>,
        hidden_ty: <Self::Interner as rustc_type_ir::Interner>::Ty,
    ) {
        todo!()
    }

    fn reset_opaque_types(&self) {
        todo!()
    }

    fn fetch_eligible_assoc_item(
        &self,
        goal_trait_ref: rustc_type_ir::TraitRef<Self::Interner>,
        trait_assoc_def_id: <Self::Interner as rustc_type_ir::Interner>::DefId,
        impl_def_id: <Self::Interner as rustc_type_ir::Interner>::DefId,
    ) -> Result<Option<<Self::Interner as rustc_type_ir::Interner>::DefId>, NoSolution> {
        todo!()
    }

    fn is_transmutable(
        &self,
        param_env: &<Self::Interner as rustc_type_ir::Interner>::ParamEnv,
        dst: <Self::Interner as rustc_type_ir::Interner>::Ty,
        src: <Self::Interner as rustc_type_ir::Interner>::Ty,
        assume: <Self::Interner as rustc_type_ir::Interner>::Const,
    ) -> Result<Certainty, NoSolution> {
        todo!()
    }

    fn evaluate_const(
        &self,
        param_env: <Self::Interner as rustc_type_ir::Interner>::ParamEnv,
        uv: rustc_type_ir::UnevaluatedConst<Self::Interner>,
    ) -> Option<<Self::Interner as rustc_type_ir::Interner>::Const> {
        todo!()
    }
}
