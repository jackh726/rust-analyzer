use hir_def::GenericDefId;
use intern::Interned;
use rustc_ast_ir::try_visit;

use crate::interner::InternedWrapper;

use super::{interned_vec, CanonicalVarInfo, DbInterner};

pub type OpaqueTypeKey = rustc_type_ir::OpaqueTypeKey<DbInterner>;
pub type PredefinedOpaquesData = rustc_type_ir::solve::PredefinedOpaquesData<DbInterner>;
pub type ExternalConstraintsData = rustc_type_ir::solve::ExternalConstraintsData<DbInterner>;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct PredefinedOpaques(Interned<InternedWrapper<PredefinedOpaquesData>>);

impl PredefinedOpaques {
    pub fn new(data: PredefinedOpaquesData) -> Self {
        PredefinedOpaques(Interned::new(InternedWrapper(data)))
    }
}

impl rustc_type_ir::visit::TypeVisitable<DbInterner> for PredefinedOpaques {
    fn visit_with<V: rustc_type_ir::visit::TypeVisitor<DbInterner>>(
        &self,
        visitor: &mut V,
    ) -> V::Result {
        self.opaque_types.visit_with(visitor)
    }
}

impl rustc_type_ir::fold::TypeFoldable<DbInterner> for PredefinedOpaques {
    fn try_fold_with<F: rustc_type_ir::fold::FallibleTypeFolder<DbInterner>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        Ok(PredefinedOpaques::new(PredefinedOpaquesData {
            opaque_types: self
                .opaque_types
                .iter()
                .cloned()
                .map(|opaque| opaque.try_fold_with(folder))
                .collect::<Result<_, F::Error>>()?,
        }))
    }
}

impl std::ops::Deref for PredefinedOpaques {
    type Target = PredefinedOpaquesData;

    fn deref(&self) -> &Self::Target {
        &self.0 .0
    }
}

interned_vec!(DefiningOpaqueTypes, GenericDefId);

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ExternalConstraints(Interned<InternedWrapper<ExternalConstraintsData>>);

impl ExternalConstraints {
    pub fn new(data: ExternalConstraintsData) -> Self {
        ExternalConstraints(Interned::new(InternedWrapper(data)))
    }
}

impl std::ops::Deref for ExternalConstraints {
    type Target = ExternalConstraintsData;

    fn deref(&self) -> &Self::Target {
        &self.0 .0
    }
}

impl rustc_type_ir::visit::TypeVisitable<DbInterner> for ExternalConstraints {
    fn visit_with<V: rustc_type_ir::visit::TypeVisitor<DbInterner>>(
        &self,
        visitor: &mut V,
    ) -> V::Result {
        try_visit!(self.region_constraints.visit_with(visitor));
        try_visit!(self.opaque_types.visit_with(visitor));
        self.normalization_nested_goals.visit_with(visitor)
    }
}

impl rustc_type_ir::fold::TypeFoldable<DbInterner> for ExternalConstraints {
    fn try_fold_with<F: rustc_type_ir::fold::FallibleTypeFolder<DbInterner>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        Ok(ExternalConstraints::new(ExternalConstraintsData {
            region_constraints: self.region_constraints.clone().try_fold_with(folder)?,
            opaque_types: self
                .opaque_types
                .iter()
                .cloned()
                .map(|opaque| opaque.try_fold_with(folder))
                .collect::<Result<_, F::Error>>()?,
            normalization_nested_goals: self
                .normalization_nested_goals
                .clone()
                .try_fold_with(folder)?,
        }))
    }
}
