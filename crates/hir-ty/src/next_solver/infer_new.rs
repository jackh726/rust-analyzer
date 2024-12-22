use std::cell::{Cell, RefCell};

use rustc_index_in_tree::IndexVec;
use rustc_type_ir::fold::TypeFoldable;
use rustc_type_ir::{CanonicalQueryInput, TypingMode};
use rustc_type_ir::{
    solve::NoSolution, ConstVid, FloatVid, InferCtxtLike, IntVid, RegionVid, TyVid, UniverseIndex,
    Variance,
};
use span::Span;

use crate::db::HirDatabase;

use super::{Const, DbInterner, DbIr, DefiningOpaqueTypes, GenericArgs, Predicate, Region, Ty};

mod ena;
mod var;
mod type_variable;

#[derive(Clone, Default)]
pub struct InferCtxtInner {
    type_variable_storage: type_variable::TypeVariableStorage<'tcx>,
    const_unification_storage: ut::UnificationTableStorage<ConstVidKey<'tcx>>,
    int_unification_storage: ut::UnificationTableStorage<ty::IntVid>,
    float_unification_storage: ut::UnificationTableStorage<ty::FloatVid>,

    //undo_log: InferCtxtUndoLogs<'tcx>,
    //projection_cache: traits::ProjectionCacheStorage<'tcx>,
    //type_variable_storage: type_variable::TypeVariableStorage<'tcx>,
    //const_unification_storage: ut::UnificationTableStorage<ConstVidKey<'tcx>>,
    //int_unification_storage: ut::UnificationTableStorage<ty::IntVid>,
    //float_unification_storage: ut::UnificationTableStorage<ty::FloatVid>,
    //region_constraint_storage: Option<RegionConstraintStorage<'tcx>>,
    //region_obligations: Vec<RegionObligation<'tcx>>,
    //opaque_type_storage: OpaqueTypeStorage<'tcx>,
}

#[derive(Clone)]
pub(crate) struct InferenceTable<'db> {
    ir: DbIr<'db>,
    typing_mode: TypingMode<DbInterner>,
    considering_regions: bool,
    skip_leak_check: bool,
    inner: RefCell<InferCtxtInner>,
    //lexical_region_resolutions: RefCell<Option<LexicalRegionResolutions<'tcx>>>,
    //selection_cache: select::SelectionCache<'tcx>,
    //evaluation_cache: select::EvaluationCache<'tcx>,
    //reported_trait_errors: RefCell<FxIndexMap<Span, (Vec<ty::Predicate<'tcx>>, ErrorGuaranteed)>>,
    //reported_signature_mismatch: RefCell<FxHashSet<(Span, Option<Span>)>>,
    //tainted_by_errors: Cell<Option<ErrorGuaranteed>>,
    universe: Cell<UniverseIndex>,
    //obligation_inspector: Cell<Option<ObligationInspector<'tcx>>>,
}

impl<'db> InferenceTable<'db> {
    pub fn new(ir: DbIr<'db>, typing_mode: TypingMode<DbInterner>) -> Self {
        InferenceTable {
            ir,
            typing_mode,
            considering_regions: true,
            skip_leak_check: false,
            inner: RefCell::new(InferCtxtInner::default()),
            //lexical_region_resolutions: RefCell::new(None),
            //selection_cache: Default::default(),
            //evaluation_cache: Default::default(),
            //reported_trait_errors: Default::default(),
            //reported_signature_mismatch: Default::default(),
            //tainted_by_errors: Cell::new(None),
            universe: Cell::new(UniverseIndex::ROOT),
            //obligation_inspector: Cell::new(None),
        }
    }

    pub fn build_with_canonical<T>(
        ir: DbIr<'db>,
        input: &CanonicalQueryInput<DbInterner, T>,
    ) -> (Self, T, CanonicalVarValues)
    where
        T: TypeFoldable<DbInterner>,
    {
        let infcx = Self::new(ir, input.typing_mode);
        let (value, args) = infcx.instantiate_canonical(span, &input.canonical);
        (infcx, value, args)
    }
}

impl<'db> InferCtxtLike for InferenceTable<'db> {
    type Ir = DbIr<'db>;
    type Interner = DbInterner;

    fn cx(&self) -> Self::Ir {
        self.ir
    }

    fn typing_mode(
        &self,
        param_env_for_debug_assertion: &<Self::Interner as rustc_type_ir::Interner>::ParamEnv,
    ) -> rustc_type_ir::TypingMode<Self::Interner> {
        self.typing_mode.clone()
    }

    fn universe(&self) -> rustc_type_ir::UniverseIndex {
        self.universe.get()
    }

    fn create_next_universe(&self) -> rustc_type_ir::UniverseIndex {
        let u = self.universe.get().next_universe();
        self.universe.set(u);
        u
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

impl<'db> InferenceTable<'db> {
    pub fn next_ty_var(&self, span: Span) -> Ty<'tcx> {
        self.next_ty_var_with_origin(TypeVariableOrigin { span, param_def_id: None })
    }

    pub fn next_ty_var_with_origin(&self, origin: TypeVariableOrigin) -> Ty<'tcx> {
        let vid = self.inner.borrow_mut().type_variables().new_var(self.universe(), origin);
        Ty::new_var(self.tcx, vid)
    }

    pub fn next_ty_var_id_in_universe(&self, span: Span, universe: ty::UniverseIndex) -> TyVid {
        let origin = TypeVariableOrigin { span, param_def_id: None };
        self.inner.borrow_mut().type_variables().new_var(universe, origin)
    }

    pub fn next_ty_var_in_universe(&self, span: Span, universe: ty::UniverseIndex) -> Ty<'tcx> {
        let vid = self.next_ty_var_id_in_universe(span, universe);
        Ty::new_var(self.tcx, vid)
    }

    pub fn next_const_var(&self, span: Span) -> ty::Const<'tcx> {
        self.next_const_var_with_origin(ConstVariableOrigin { span, param_def_id: None })
    }

    pub fn next_const_var_with_origin(&self, origin: ConstVariableOrigin) -> ty::Const<'tcx> {
        let vid = self
            .inner
            .borrow_mut()
            .const_unification_table()
            .new_key(ConstVariableValue::Unknown { origin, universe: self.universe() })
            .vid;
        ty::Const::new_var(self.tcx, vid)
    }

    pub fn next_const_var_in_universe(
        &self,
        span: Span,
        universe: ty::UniverseIndex,
    ) -> ty::Const<'tcx> {
        let origin = ConstVariableOrigin { span, param_def_id: None };
        let vid = self
            .inner
            .borrow_mut()
            .const_unification_table()
            .new_key(ConstVariableValue::Unknown { origin, universe })
            .vid;
        ty::Const::new_var(self.tcx, vid)
    }

    pub fn next_int_var(&self) -> Ty<'tcx> {
        let next_int_var_id =
            self.inner.borrow_mut().int_unification_table().new_key(ty::IntVarValue::Unknown);
        Ty::new_int_var(self.tcx, next_int_var_id)
    }

    pub fn next_float_var(&self) -> Ty<'tcx> {
        let next_float_var_id =
            self.inner.borrow_mut().float_unification_table().new_key(ty::FloatVarValue::Unknown);
        Ty::new_float_var(self.tcx, next_float_var_id)
    }

    pub fn next_region_var(&self, origin: RegionVariableOrigin) -> ty::Region<'tcx> {
        self.next_region_var_in_universe(origin, self.universe())
    }

    pub fn next_region_var_in_universe(
        &self,
        origin: RegionVariableOrigin,
        universe: ty::UniverseIndex,
    ) -> ty::Region<'tcx> {
        let region_var =
            self.inner.borrow_mut().unwrap_region_constraints().new_region_var(universe, origin);
        ty::Region::new_var(self.tcx, region_var)
    }
}

impl<'db> InferenceTable<'db> {
    pub fn instantiate_canonical<T>(
        &self,
        span: Span,
        canonical: &Canonical<T>,
    ) -> (T, CanonicalVarValues)
    where
        T: TypeFoldable<DbInterner>,
    {
        let universes: IndexVec<ty::UniverseIndex, _> = std::iter::once(self.universe())
            .chain((1..=canonical.max_universe.as_u32()).map(|_| self.create_next_universe()))
            .collect();

        let canonical_inference_vars =
            self.instantiate_canonical_vars(span, canonical.variables, |ui| universes[ui]);
        let result = canonical.instantiate(self.tcx, &canonical_inference_vars);
        (result, canonical_inference_vars)
    }

    fn instantiate_canonical_vars(
        &self,
        span: Span,
        variables: &List<CanonicalVarInfo<'tcx>>,
        universe_map: impl Fn(ty::UniverseIndex) -> ty::UniverseIndex,
    ) -> CanonicalVarValues<'tcx> {
        CanonicalVarValues {
            var_values: self.tcx.mk_args_from_iter(
                variables
                    .iter()
                    .map(|info| self.instantiate_canonical_var(span, info, &universe_map)),
            ),
        }
    }

    pub fn instantiate_canonical_var(
        &self,
        span: Span,
        cv_info: CanonicalVarInfo,
        universe_map: impl Fn(ty::UniverseIndex) -> ty::UniverseIndex,
    ) -> GenericArg {
        match cv_info.kind {
            CanonicalVarKind::Ty(ty_kind) => {
                let ty = match ty_kind {
                    CanonicalTyVarKind::General(ui) => {
                        self.next_ty_var_in_universe(span, universe_map(ui))
                    }

                    CanonicalTyVarKind::Int => self.next_int_var(),

                    CanonicalTyVarKind::Float => self.next_float_var(),
                };
                ty.into()
            }

            CanonicalVarKind::PlaceholderTy(ty::PlaceholderType { universe, bound }) => {
                let universe_mapped = universe_map(universe);
                let placeholder_mapped = ty::PlaceholderType { universe: universe_mapped, bound };
                Ty::new_placeholder(self.tcx, placeholder_mapped).into()
            }

            CanonicalVarKind::Region(ui) => self
                .next_region_var_in_universe(
                    RegionVariableOrigin::MiscVariable(span),
                    universe_map(ui),
                )
                .into(),

            CanonicalVarKind::PlaceholderRegion(ty::PlaceholderRegion { universe, bound }) => {
                let universe_mapped = universe_map(universe);
                let placeholder_mapped = ty::PlaceholderRegion { universe: universe_mapped, bound };
                ty::Region::new_placeholder(self.tcx, placeholder_mapped).into()
            }

            CanonicalVarKind::Const(ui) => {
                self.next_const_var_in_universe(span, universe_map(ui)).into()
            }
            CanonicalVarKind::PlaceholderConst(ty::PlaceholderConst { universe, bound }) => {
                let universe_mapped = universe_map(universe);
                let placeholder_mapped = ty::PlaceholderConst { universe: universe_mapped, bound };
                ty::Const::new_placeholder(self.tcx, placeholder_mapped).into()
            }
        }
    }
}
