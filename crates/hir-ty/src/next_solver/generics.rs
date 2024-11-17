use hir_def::{GenericDefId, GenericParamId};

use crate::db::HirDatabase;

use super::{
    interner::Ty,
    mapping::{const_to_param_idx, lt_to_param_idx, ty_to_param_idx},
    Const, Region,
};

use super::{DbInterner, GenericArg};

#[derive(Debug)]
pub struct Generics {
    pub parent: Option<GenericDefId>,

    pub own_params: Vec<GenericParamDef>,
}

#[derive(Debug)]
pub struct GenericParamDef {
    pub def_id: GenericDefId,
    pub index: u32,

    pub kind: GenericParamDefKind,
}

#[derive(Debug)]
pub enum GenericParamDefKind {
    Lifetime,
    Type,
    Const,
}

impl rustc_type_ir::inherent::GenericsOf<DbInterner> for Generics {
    fn count(&self) -> usize {
        todo!()
    }
}

impl GenericParamDef {
    pub fn to_error(&self, interner: DbInterner) -> GenericArg {
        todo!()
    }
}

impl DbInterner {
    pub fn mk_param_from_def(self, param: &GenericParamDef) -> GenericArg {
        todo!()
    }
}

/*
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
*/
