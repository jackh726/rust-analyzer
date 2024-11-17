use hir_def::GenericParamId;

use crate::{db::HirDatabase, generics::Generics};

use super::{
    interner::{Const, GenericArgs, Region, Ty},
    mapping::{const_to_param_idx, lt_to_param_idx, ty_to_param_idx},
};

impl Generics {
    /// Returns a Substitution that replaces each parameter by itself (i.e. `Ty::Param`).
    pub(crate) fn rustc_param_subst(&self, db: &dyn HirDatabase) -> GenericArgs {
        GenericArgs::new(self.iter_id().map(|id| match id {
            GenericParamId::TypeParamId(id) => {
                let kind = rustc_type_ir::TyKind::Param(ty_to_param_idx(db, id.into()));
                let ty = Ty::new(kind);
                ty.into()
            }
            GenericParamId::ConstParamId(id) => {
                let kind = rustc_type_ir::ConstKind::Param(const_to_param_idx(db, id.into()));
                let ct = Const::new(kind);
                ct.into()
            }
            GenericParamId::LifetimeParamId(id) => {
                let kind = rustc_type_ir::RegionKind::ReEarlyParam(lt_to_param_idx(db, id.into()));
                let lt = Region::new(kind);
                lt.into()
            }
        }))
    }
}
