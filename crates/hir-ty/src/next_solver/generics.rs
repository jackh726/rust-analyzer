use hir_def::GenericParamId;

use crate::{db::HirDatabase, generics::Generics};

use super::{
    interner::{RustcConst, RustcGenericArgs, RustcRegion, RustcTy},
    mapping::{const_to_param_idx, lt_to_param_idx, ty_to_param_idx},
};

impl Generics {
    /// Returns a Substitution that replaces each parameter by itself (i.e. `Ty::Param`).
    pub(crate) fn rustc_param_subst(&self, db: &dyn HirDatabase) -> RustcGenericArgs {
        RustcGenericArgs::new(self.iter_id().map(|id| match id {
            GenericParamId::TypeParamId(id) => {
                let kind = rustc_type_ir::TyKind::Param(ty_to_param_idx(db, id.into()));
                let ty = RustcTy::new(kind);
                ty.into()
            }
            GenericParamId::ConstParamId(id) => {
                let kind = rustc_type_ir::ConstKind::Param(const_to_param_idx(db, id.into()));
                let ct = RustcConst::new(kind);
                ct.into()
            }
            GenericParamId::LifetimeParamId(id) => {
                let kind = rustc_type_ir::RegionKind::ReEarlyParam(lt_to_param_idx(db, id.into()));
                let lt = RustcRegion::new(kind);
                lt.into()
            }
        }))
    }
}
