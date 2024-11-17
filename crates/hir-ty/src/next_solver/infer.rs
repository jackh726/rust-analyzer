use rustc_type_ir::{
    solve::NoSolution, ConstVid, FloatVid, InferCtxtLike, IntVid, RegionVid, TyVid, UniverseIndex,
    Variance,
};

use crate::db::HirDatabase;

use super::{
    Const, DbInterner, DbIr, DefiningOpaqueTypes, GenericArgs, ParamEnv, Predicate, Region, Ty,
};

pub type Binder<T> = rustc_type_ir::Binder<DbInterner, T>;
pub type Canonical<T> = rustc_type_ir::Canonical<DbInterner, T>;
pub type CanonicalVarValues = rustc_type_ir::CanonicalVarValues<DbInterner>;
pub type CanonicalVarInfo = rustc_type_ir::CanonicalVarInfo<DbInterner>;

#[derive(Clone)]
pub(crate) struct InferenceTable<'db> {
    interner: DbIr<'db>,
    pub(crate) db: &'db dyn HirDatabase,
    // unify: ena::unify::InPlaceUnificationTable<TyVid>,
    // vars: Vec<TyVid>,
    max_universe: UniverseIndex,
}

impl<'db> InferCtxtLike for InferenceTable<'db> {
    type Ir = DbIr<'db>;
    type Interner = DbInterner;

    fn cx(&self) -> Self::Ir {
        self.interner
    }

    fn typing_mode(
        &self,
        param_env_for_debug_assertion: &<Self::Interner as rustc_type_ir::Interner>::ParamEnv,
    ) -> rustc_type_ir::TypingMode<Self::Interner> {
        todo!()
    }

    fn universe(&self) -> rustc_type_ir::UniverseIndex {
        todo!()
    }

    fn create_next_universe(&self) -> rustc_type_ir::UniverseIndex {
        todo!()
    }

    fn universe_of_ty(&self, ty: rustc_type_ir::TyVid) -> Option<rustc_type_ir::UniverseIndex> {
        todo!()
    }

    fn universe_of_lt(&self, lt: rustc_type_ir::RegionVid) -> Option<rustc_type_ir::UniverseIndex> {
        todo!()
    }

    fn universe_of_ct(&self, ct: rustc_type_ir::ConstVid) -> Option<rustc_type_ir::UniverseIndex> {
        todo!()
    }

    fn root_ty_var(&self, var: rustc_type_ir::TyVid) -> rustc_type_ir::TyVid {
        todo!()
    }

    fn root_const_var(&self, var: rustc_type_ir::ConstVid) -> rustc_type_ir::ConstVid {
        todo!()
    }

    fn opportunistic_resolve_ty_var(
        &self,
        vid: rustc_type_ir::TyVid,
    ) -> <Self::Interner as rustc_type_ir::Interner>::Ty {
        todo!()
    }

    fn opportunistic_resolve_int_var(
        &self,
        vid: rustc_type_ir::IntVid,
    ) -> <Self::Interner as rustc_type_ir::Interner>::Ty {
        todo!()
    }

    fn opportunistic_resolve_float_var(
        &self,
        vid: rustc_type_ir::FloatVid,
    ) -> <Self::Interner as rustc_type_ir::Interner>::Ty {
        todo!()
    }

    fn opportunistic_resolve_ct_var(
        &self,
        vid: rustc_type_ir::ConstVid,
    ) -> <Self::Interner as rustc_type_ir::Interner>::Const {
        todo!()
    }

    fn opportunistic_resolve_lt_var(
        &self,
        vid: rustc_type_ir::RegionVid,
    ) -> <Self::Interner as rustc_type_ir::Interner>::Region {
        todo!()
    }

    fn next_ty_infer(&self) -> <Self::Interner as rustc_type_ir::Interner>::Ty {
        todo!()
    }

    fn next_const_infer(&self) -> <Self::Interner as rustc_type_ir::Interner>::Const {
        todo!()
    }

    fn fresh_args_for_item(
        &self,
        def_id: <Self::Interner as rustc_type_ir::Interner>::DefId,
    ) -> <Self::Interner as rustc_type_ir::Interner>::GenericArgs {
        todo!()
    }

    fn instantiate_binder_with_infer<T: rustc_type_ir::fold::TypeFoldable<Self::Interner>>(
        &self,
        value: rustc_type_ir::Binder<Self::Interner, T>,
    ) -> T {
        todo!()
    }

    fn enter_forall<T: rustc_type_ir::fold::TypeFoldable<Self::Interner>, U>(
        &self,
        value: rustc_type_ir::Binder<Self::Interner, T>,
        f: impl FnOnce(T) -> U,
    ) -> U {
        todo!()
    }

    fn equate_ty_vids_raw(&self, a: rustc_type_ir::TyVid, b: rustc_type_ir::TyVid) {
        todo!()
    }

    fn equate_int_vids_raw(&self, a: rustc_type_ir::IntVid, b: rustc_type_ir::IntVid) {
        todo!()
    }

    fn equate_float_vids_raw(&self, a: rustc_type_ir::FloatVid, b: rustc_type_ir::FloatVid) {
        todo!()
    }

    fn equate_const_vids_raw(&self, a: rustc_type_ir::ConstVid, b: rustc_type_ir::ConstVid) {
        todo!()
    }

    fn instantiate_ty_var_raw<
        R: rustc_type_ir::relate::combine::PredicateEmittingRelation<Self>,
    >(
        &self,
        relation: &mut R,
        target_is_expected: bool,
        target_vid: rustc_type_ir::TyVid,
        instantiation_variance: rustc_type_ir::Variance,
        source_ty: <Self::Interner as rustc_type_ir::Interner>::Ty,
    ) -> rustc_type_ir::relate::RelateResult<Self::Interner, ()> {
        todo!()
    }

    fn instantiate_int_var_raw(
        &self,
        vid: rustc_type_ir::IntVid,
        value: rustc_type_ir::IntVarValue,
    ) {
        todo!()
    }

    fn instantiate_float_var_raw(
        &self,
        vid: rustc_type_ir::FloatVid,
        value: rustc_type_ir::FloatVarValue,
    ) {
        todo!()
    }

    fn instantiate_const_var_raw<
        R: rustc_type_ir::relate::combine::PredicateEmittingRelation<Self>,
    >(
        &self,
        relation: &mut R,
        target_is_expected: bool,
        target_vid: rustc_type_ir::ConstVid,
        source_ct: <Self::Interner as rustc_type_ir::Interner>::Const,
    ) -> rustc_type_ir::relate::RelateResult<Self::Interner, ()> {
        todo!()
    }

    fn set_tainted_by_errors(
        &self,
        e: <Self::Interner as rustc_type_ir::Interner>::ErrorGuaranteed,
    ) {
        todo!()
    }

    fn shallow_resolve(
        &self,
        ty: <Self::Interner as rustc_type_ir::Interner>::Ty,
    ) -> <Self::Interner as rustc_type_ir::Interner>::Ty {
        todo!()
    }

    fn shallow_resolve_const(
        &self,
        ty: <Self::Interner as rustc_type_ir::Interner>::Const,
    ) -> <Self::Interner as rustc_type_ir::Interner>::Const {
        todo!()
    }

    fn resolve_vars_if_possible<T>(&self, value: T) -> T
    where
        T: rustc_type_ir::fold::TypeFoldable<Self::Interner>,
    {
        todo!()
    }

    fn probe<T>(&self, probe: impl FnOnce() -> T) -> T {
        todo!()
    }

    fn sub_regions(
        &self,
        sub: <Self::Interner as rustc_type_ir::Interner>::Region,
        sup: <Self::Interner as rustc_type_ir::Interner>::Region,
    ) {
        todo!()
    }

    fn equate_regions(
        &self,
        a: <Self::Interner as rustc_type_ir::Interner>::Region,
        b: <Self::Interner as rustc_type_ir::Interner>::Region,
    ) {
        todo!()
    }

    fn register_ty_outlives(
        &self,
        ty: <Self::Interner as rustc_type_ir::Interner>::Ty,
        r: <Self::Interner as rustc_type_ir::Interner>::Region,
    ) {
        todo!()
    }
}
