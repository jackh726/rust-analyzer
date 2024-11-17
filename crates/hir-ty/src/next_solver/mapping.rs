use base_db::ra_salsa;
use chalk_ir::interner::HasInterner;
use hir_def::{ConstParamId, FunctionId, LifetimeParamId, TypeAliasId, TypeParamId};
use rustc_type_ir::{
    fold::{TypeFoldable, TypeSuperFoldable},
    inherent::{BoundVarLike, IntoKind, PlaceholderLike, SliceLike},
    visit::TypeVisitable,
};

use crate::{
    db::HirDatabase,
    next_solver::interner::{
        RustcAbi, RustcAdtDef, RustcBoundConst, RustcBoundExistentialPredicates, RustcBoundRegion,
        RustcBoundTy, RustcBoundVarKind, RustcBoundVarKinds, RustcConst, RustcEarlyParamRegion,
        RustcErrorGuaranteed, RustcGenericArg, RustcGenericArgs, RustcInterner, RustcParamConst,
        RustcParamTy, RustcPlaceholderConst, RustcPlaceholderRegion, RustcPlaceholderTy,
        RustcRegion, RustcSafety, RustcTy, RustcTys, RustcValueConst,
    },
    Interner,
};

use super::interner::{BoundRegionKind, BoundTyKind};

pub fn ty_to_param_idx(db: &dyn HirDatabase, id: TypeParamId) -> RustcParamTy {
    let interned_id = db.intern_type_or_const_param_id(id.into());
    RustcParamTy { index: ra_salsa::InternKey::as_intern_id(&interned_id).as_u32() }
}

pub fn const_to_param_idx(db: &dyn HirDatabase, id: ConstParamId) -> RustcParamConst {
    let interned_id = db.intern_type_or_const_param_id(id.into());
    RustcParamConst { index: ra_salsa::InternKey::as_intern_id(&interned_id).as_u32() }
}

pub fn lt_to_param_idx(db: &dyn HirDatabase, id: LifetimeParamId) -> RustcEarlyParamRegion {
    let interned_id = db.intern_lifetime_param_id(id);
    RustcEarlyParamRegion { index: ra_salsa::InternKey::as_intern_id(&interned_id).as_u32() }
}

pub fn convert_binder_to_early_binder<T: rustc_type_ir::fold::TypeFoldable<RustcInterner>>(
    binder: rustc_type_ir::Binder<RustcInterner, T>,
) -> rustc_type_ir::EarlyBinder<RustcInterner, T> {
    let mut folder = BinderToEarlyBinder { debruijn: rustc_type_ir::DebruijnIndex::ZERO };
    rustc_type_ir::EarlyBinder::bind(binder.skip_binder().fold_with(&mut folder))
}

struct BinderToEarlyBinder {
    debruijn: rustc_type_ir::DebruijnIndex,
}

impl rustc_type_ir::fold::TypeFolder<RustcInterner> for BinderToEarlyBinder {
    fn cx(&self) -> RustcInterner {
        RustcInterner
    }

    fn fold_binder<T>(
        &mut self,
        t: rustc_type_ir::Binder<RustcInterner, T>,
    ) -> rustc_type_ir::Binder<RustcInterner, T>
    where
        T: TypeFoldable<RustcInterner>,
    {
        self.debruijn.shift_in(1);
        let result = t.super_fold_with(self);
        self.debruijn.shift_out(1);
        result
    }

    fn fold_ty(&mut self, t: RustcTy) -> RustcTy {
        match t.clone().kind() {
            rustc_type_ir::TyKind::Bound(debruijn, bound_ty) if self.debruijn == debruijn => {
                let var: rustc_type_ir::BoundVar = bound_ty.var();
                RustcTy::new(rustc_type_ir::TyKind::Param(RustcParamTy { index: var.as_u32() }))
            }
            _ => t,
        }
    }

    fn fold_region(&mut self, r: RustcRegion) -> RustcRegion {
        match r.clone().kind() {
            rustc_type_ir::ReBound(debruijn, bound_region) if self.debruijn == debruijn => {
                let var: rustc_type_ir::BoundVar = bound_region.var();
                RustcRegion::new(rustc_type_ir::RegionKind::ReEarlyParam(RustcEarlyParamRegion {
                    index: var.as_u32(),
                }))
            }
            _ => r,
        }
    }

    fn fold_const(&mut self, c: RustcConst) -> RustcConst {
        match c.clone().kind() {
            rustc_type_ir::ConstKind::Bound(debruijn, bound_const) if self.debruijn == debruijn => {
                let var: rustc_type_ir::BoundVar = bound_const.var();
                RustcConst::new(rustc_type_ir::ConstKind::Param(RustcParamConst {
                    index: var.as_u32(),
                }))
            }
            _ => c,
        }
    }
}

pub trait ChalkToRustc<Rustc> {
    fn to_rustc(&self) -> Rustc;
}

impl ChalkToRustc<RustcTy> for chalk_ir::Ty<Interner> {
    fn to_rustc(&self) -> RustcTy {
        RustcTy::new(match self.kind(Interner) {
            chalk_ir::TyKind::Adt(adt_id, substitution) => {
                let def = RustcAdtDef::new(adt_id.0);
                let args = substitution.to_rustc();
                rustc_type_ir::TyKind::Adt(def, args)
            }
            chalk_ir::TyKind::AssociatedType(assoc_type_id, substitution) => {
                let id: TypeAliasId = ra_salsa::InternKey::from_intern_id(assoc_type_id.0);
                let args: RustcGenericArgs = substitution.to_rustc();
                let alias_ty = rustc_type_ir::AliasTy::new(RustcInterner, id.into(), args.iter());
                rustc_type_ir::TyKind::Alias(rustc_type_ir::AliasTyKind::Opaque, alias_ty)
            }
            chalk_ir::TyKind::Scalar(scalar) => match scalar {
                chalk_ir::Scalar::Bool => rustc_type_ir::TyKind::Bool,
                chalk_ir::Scalar::Char => rustc_type_ir::TyKind::Char,
                chalk_ir::Scalar::Int(chalk_ir::IntTy::Isize) => {
                    rustc_type_ir::TyKind::Int(rustc_type_ir::IntTy::Isize)
                }
                chalk_ir::Scalar::Int(chalk_ir::IntTy::I8) => {
                    rustc_type_ir::TyKind::Int(rustc_type_ir::IntTy::I8)
                }
                chalk_ir::Scalar::Int(chalk_ir::IntTy::I16) => {
                    rustc_type_ir::TyKind::Int(rustc_type_ir::IntTy::I16)
                }
                chalk_ir::Scalar::Int(chalk_ir::IntTy::I32) => {
                    rustc_type_ir::TyKind::Int(rustc_type_ir::IntTy::I32)
                }
                chalk_ir::Scalar::Int(chalk_ir::IntTy::I64) => {
                    rustc_type_ir::TyKind::Int(rustc_type_ir::IntTy::I64)
                }
                chalk_ir::Scalar::Int(chalk_ir::IntTy::I128) => {
                    rustc_type_ir::TyKind::Int(rustc_type_ir::IntTy::I128)
                }
                chalk_ir::Scalar::Uint(chalk_ir::UintTy::Usize) => {
                    rustc_type_ir::TyKind::Uint(rustc_type_ir::UintTy::Usize)
                }
                chalk_ir::Scalar::Uint(chalk_ir::UintTy::U8) => {
                    rustc_type_ir::TyKind::Uint(rustc_type_ir::UintTy::U8)
                }
                chalk_ir::Scalar::Uint(chalk_ir::UintTy::U16) => {
                    rustc_type_ir::TyKind::Uint(rustc_type_ir::UintTy::U16)
                }
                chalk_ir::Scalar::Uint(chalk_ir::UintTy::U32) => {
                    rustc_type_ir::TyKind::Uint(rustc_type_ir::UintTy::U32)
                }
                chalk_ir::Scalar::Uint(chalk_ir::UintTy::U64) => {
                    rustc_type_ir::TyKind::Uint(rustc_type_ir::UintTy::U64)
                }
                chalk_ir::Scalar::Uint(chalk_ir::UintTy::U128) => {
                    rustc_type_ir::TyKind::Uint(rustc_type_ir::UintTy::U128)
                }
                chalk_ir::Scalar::Float(chalk_ir::FloatTy::F16) => {
                    rustc_type_ir::TyKind::Float(rustc_type_ir::FloatTy::F16)
                }
                chalk_ir::Scalar::Float(chalk_ir::FloatTy::F32) => {
                    rustc_type_ir::TyKind::Float(rustc_type_ir::FloatTy::F32)
                }
                chalk_ir::Scalar::Float(chalk_ir::FloatTy::F64) => {
                    rustc_type_ir::TyKind::Float(rustc_type_ir::FloatTy::F64)
                }
                chalk_ir::Scalar::Float(chalk_ir::FloatTy::F128) => {
                    rustc_type_ir::TyKind::Float(rustc_type_ir::FloatTy::F128)
                }
            },
            chalk_ir::TyKind::Tuple(_, substitution) => {
                let args = substitution.to_rustc();
                rustc_type_ir::TyKind::Tuple(args)
            }
            chalk_ir::TyKind::Array(ty, len) => {
                rustc_type_ir::TyKind::Array(ty.to_rustc(), len.to_rustc())
            }
            chalk_ir::TyKind::Slice(ty) => rustc_type_ir::TyKind::Slice(ty.to_rustc()),
            chalk_ir::TyKind::Raw(mutability, ty) => {
                rustc_type_ir::RawPtr(ty.to_rustc(), mutability.to_rustc())
            }
            chalk_ir::TyKind::Ref(mutability, lifetime, ty) => rustc_type_ir::TyKind::Ref(
                lifetime.to_rustc(),
                ty.to_rustc(),
                mutability.to_rustc(),
            ),
            chalk_ir::TyKind::OpaqueType(_, _) => {
                //let impl_trait_id = db.lookup_intern_impl_trait_id(id);
                //let alias_ty = rustc_type_ir::AliasTy::new(RustcInterner, def_id.into(), substitution.to_rustc());
                //rustc_type_ir::TyKind::Alias(rustc_type_ir::AliasTyKind::Opaque, alias_ty)
                todo!("Needs GenericDefId::ImplTraitId")
            }
            chalk_ir::TyKind::FnDef(fn_def_id, substitution) => {
                let id: FunctionId = ra_salsa::InternKey::from_intern_id(fn_def_id.0);
                rustc_type_ir::TyKind::FnDef(id.into(), substitution.to_rustc())
            }
            chalk_ir::TyKind::Str => rustc_type_ir::TyKind::Str,
            chalk_ir::TyKind::Never => rustc_type_ir::TyKind::Never,
            chalk_ir::TyKind::Closure(_closure_id, _substitution) => {
                //let id = ra_salsa::InternKey::from_intern_id(closure_id.0);
                //rustc_type_ir::TyKind::Closure(id.into(), substitution.to_rustc())
                todo!("Needs GenericDefId::Closure")
            }
            chalk_ir::TyKind::Coroutine(_coroutine_id, _substitution) => {
                //let id = ra_salsa::InternKey::from_intern_id(coroutine_id.0);
                //rustc_type_ir::TyKind::Coroutine(id.into(), substitution.to_rustc())
                todo!("Needs GenericDefId::Coroutine")
            }
            chalk_ir::TyKind::CoroutineWitness(_coroutine_id, _substitution) => {
                //let id = ra_salsa::InternKey::from_intern_id(coroutine_id.0);
                //rustc_type_ir::TyKind::CoroutineWitness(id.into(), substitution.to_rustc())
                todo!("Needs GenericDefId::Coroutine")
            }
            chalk_ir::TyKind::Foreign(_foreign_def_id) => {
                //let id = ra_salsa::InternKey::from_intern_id(foreign_def_id.0);
                //rustc_type_ir::TyKind::Foreign(id.into())
                todo!("Needs GenericDefId::Foreign")
            }
            chalk_ir::TyKind::Error => rustc_type_ir::TyKind::Error(RustcErrorGuaranteed),
            chalk_ir::TyKind::Placeholder(placeholder_index) => {
                rustc_type_ir::TyKind::Placeholder(RustcPlaceholderTy::new(
                    placeholder_index.ui.to_rustc(),
                    rustc_type_ir::BoundVar::from_usize(placeholder_index.idx),
                ))
            }
            chalk_ir::TyKind::Dyn(dyn_ty) => {
                let bounds = dyn_ty.bounds.to_rustc();
                let region = dyn_ty.lifetime.to_rustc();
                let kind = rustc_type_ir::DynKind::Dyn;
                rustc_type_ir::TyKind::Dynamic(bounds, region, kind)
            }
            chalk_ir::TyKind::Alias(alias_ty) => {
                match alias_ty {
                    chalk_ir::AliasTy::Projection(projection_ty) => {
                        let def_id: TypeAliasId =
                            ra_salsa::InternKey::from_intern_id(projection_ty.associated_ty_id.0);
                        let alias_ty = rustc_type_ir::AliasTy::new_from_args(
                            RustcInterner,
                            def_id.into(),
                            projection_ty.substitution.to_rustc(),
                        );
                        rustc_type_ir::TyKind::Alias(
                            rustc_type_ir::AliasTyKind::Projection,
                            alias_ty,
                        )
                    }
                    chalk_ir::AliasTy::Opaque(_opaque_ty) => {
                        //let def_id: TypeAliasId = ra_salsa::InternKey::from_intern_id(opaque_ty.opaque_ty_id);
                        //let alias_ty = rustc_type_ir::AliasTy::new(RustcInterner, def_id.into(), opaque_ty.substitution.to_rustc());
                        //rustc_type_ir::TyKind::Alias(rustc_type_ir::AliasTyKind::Opaque, alias_ty)
                        todo!("Needs GenericDefId::ImplTraitId")
                    }
                }
            }
            chalk_ir::TyKind::Function(fn_pointer) => {
                let sig_tys = fn_pointer.clone().into_binders(Interner).to_rustc();
                let header = rustc_type_ir::FnHeader {
                    abi: RustcAbi::new(fn_pointer.sig.abi),
                    c_variadic: fn_pointer.sig.variadic,
                    safety: RustcSafety::new(fn_pointer.sig.safety),
                };

                rustc_type_ir::TyKind::FnPtr(sig_tys, header)
            }
            chalk_ir::TyKind::BoundVar(bound_var) => rustc_type_ir::TyKind::Bound(
                bound_var.debruijn.to_rustc(),
                RustcBoundTy::new(rustc_type_ir::BoundVar::from_usize(bound_var.index)),
            ),
            chalk_ir::TyKind::InferenceVar(inference_var, ty_variable_kind) => {
                rustc_type_ir::TyKind::Infer(
                    (inference_var.clone(), ty_variable_kind.clone()).to_rustc(),
                )
            }
        })
    }
}

impl ChalkToRustc<RustcRegion> for chalk_ir::Lifetime<Interner> {
    fn to_rustc(&self) -> RustcRegion {
        RustcRegion::new(match self.data(Interner) {
            chalk_ir::LifetimeData::BoundVar(bound_var) => rustc_type_ir::RegionKind::ReBound(
                bound_var.debruijn.to_rustc(),
                RustcBoundRegion::new(rustc_type_ir::BoundVar::from_u32(bound_var.index as u32)),
            ),
            chalk_ir::LifetimeData::InferenceVar(inference_var) => {
                rustc_type_ir::RegionKind::ReVar(rustc_type_ir::RegionVid::from_u32(
                    inference_var.index(),
                ))
            }
            chalk_ir::LifetimeData::Placeholder(placeholder_index) => {
                rustc_type_ir::RegionKind::RePlaceholder(RustcPlaceholderRegion::new(
                    rustc_type_ir::UniverseIndex::from_u32(placeholder_index.ui.counter as u32),
                    rustc_type_ir::BoundVar::from_u32(placeholder_index.idx as u32),
                ))
            }
            chalk_ir::LifetimeData::Static => rustc_type_ir::RegionKind::ReStatic,
            chalk_ir::LifetimeData::Erased => rustc_type_ir::RegionKind::ReErased,
            chalk_ir::LifetimeData::Phantom(_, _) => {
                unreachable!()
            }
            chalk_ir::LifetimeData::Error => {
                rustc_type_ir::RegionKind::ReError(RustcErrorGuaranteed)
            }
        })
    }
}

impl ChalkToRustc<RustcConst> for chalk_ir::Const<Interner> {
    fn to_rustc(&self) -> RustcConst {
        let data = self.data(Interner);
        RustcConst::new(match &data.value {
            chalk_ir::ConstValue::BoundVar(bound_var) => rustc_type_ir::ConstKind::Bound(
                bound_var.debruijn.to_rustc(),
                RustcBoundConst::new(rustc_type_ir::BoundVar::from_usize(bound_var.index)),
            ),
            chalk_ir::ConstValue::InferenceVar(inference_var) => {
                rustc_type_ir::ConstKind::Infer(rustc_type_ir::InferConst::Var(
                    rustc_type_ir::ConstVid::from_u32(inference_var.index()),
                ))
            }
            chalk_ir::ConstValue::Placeholder(placeholder_index) => {
                rustc_type_ir::ConstKind::Placeholder(RustcPlaceholderConst::new(
                    placeholder_index.ui.to_rustc(),
                    rustc_type_ir::BoundVar::from_usize(placeholder_index.idx),
                ))
            }
            chalk_ir::ConstValue::Concrete(concrete_const) => rustc_type_ir::ConstKind::Value(
                data.ty.to_rustc(),
                RustcValueConst::new(concrete_const.interned.clone()),
            ),
        })
    }
}

impl ChalkToRustc<RustcBoundExistentialPredicates>
    for chalk_ir::Binders<chalk_ir::QuantifiedWhereClauses<Interner>>
{
    fn to_rustc(&self) -> RustcBoundExistentialPredicates {
        todo!()
    }
}

impl ChalkToRustc<rustc_type_ir::FnSigTys<RustcInterner>> for chalk_ir::FnSubst<Interner> {
    fn to_rustc(&self) -> rustc_type_ir::FnSigTys<RustcInterner> {
        todo!()
    }
}

impl<
        U: TypeVisitable<RustcInterner>,
        T: Clone + ChalkToRustc<U> + HasInterner<Interner = Interner>,
    > ChalkToRustc<rustc_type_ir::Binder<RustcInterner, U>> for chalk_ir::Binders<T>
{
    fn to_rustc(&self) -> rustc_type_ir::Binder<RustcInterner, U> {
        let (val, binders) = self.clone().into_value_and_skipped_binders();
        rustc_type_ir::Binder::bind_with_vars(val.to_rustc(), binders.to_rustc())
    }
}

impl ChalkToRustc<RustcBoundVarKinds> for chalk_ir::VariableKinds<Interner> {
    fn to_rustc(&self) -> RustcBoundVarKinds {
        RustcBoundVarKinds::new(self.iter(Interner).map(|v| v.to_rustc()))
    }
}

impl ChalkToRustc<RustcBoundVarKind> for chalk_ir::VariableKind<Interner> {
    fn to_rustc(&self) -> RustcBoundVarKind {
        match self {
            chalk_ir::VariableKind::Ty(_ty_variable_kind) => {
                RustcBoundVarKind::Ty(BoundTyKind::Anon)
            }
            chalk_ir::VariableKind::Lifetime => RustcBoundVarKind::Region(BoundRegionKind::Anon),
            chalk_ir::VariableKind::Const(_ty) => RustcBoundVarKind::Const,
        }
    }
}

impl ChalkToRustc<RustcGenericArgs> for chalk_ir::Substitution<Interner> {
    fn to_rustc(&self) -> RustcGenericArgs {
        RustcGenericArgs::new(self.iter(Interner).map(|arg| -> RustcGenericArg {
            match arg.data(Interner) {
                chalk_ir::GenericArgData::Ty(ty) => ty.to_rustc().into(),
                chalk_ir::GenericArgData::Lifetime(lifetime) => lifetime.to_rustc().into(),
                chalk_ir::GenericArgData::Const(_) => todo!(),
            }
        }))
    }
}

impl ChalkToRustc<RustcTys> for chalk_ir::Substitution<Interner> {
    fn to_rustc(&self) -> RustcTys {
        RustcTys::new(self.iter(Interner).map(|arg| -> RustcTy {
            match arg.data(Interner) {
                chalk_ir::GenericArgData::Ty(ty) => ty.to_rustc(),
                chalk_ir::GenericArgData::Lifetime(_) => unreachable!(),
                chalk_ir::GenericArgData::Const(_) => unreachable!(),
            }
        }))
    }
}

impl ChalkToRustc<rustc_type_ir::DebruijnIndex> for chalk_ir::DebruijnIndex {
    fn to_rustc(&self) -> rustc_type_ir::DebruijnIndex {
        rustc_type_ir::DebruijnIndex::from_u32(self.depth())
    }
}

impl ChalkToRustc<rustc_type_ir::UniverseIndex> for chalk_ir::UniverseIndex {
    fn to_rustc(&self) -> rustc_type_ir::UniverseIndex {
        rustc_type_ir::UniverseIndex::from_u32(self.counter as u32)
    }
}

impl ChalkToRustc<rustc_type_ir::InferTy> for (chalk_ir::InferenceVar, chalk_ir::TyVariableKind) {
    fn to_rustc(&self) -> rustc_type_ir::InferTy {
        match self.1 {
            chalk_ir::TyVariableKind::General => {
                rustc_type_ir::InferTy::TyVar(rustc_type_ir::TyVid::from_u32(self.0.index()))
            }
            chalk_ir::TyVariableKind::Integer => {
                rustc_type_ir::InferTy::IntVar(rustc_type_ir::IntVid::from_u32(self.0.index()))
            }
            chalk_ir::TyVariableKind::Float => {
                rustc_type_ir::InferTy::FloatVar(rustc_type_ir::FloatVid::from_u32(self.0.index()))
            }
        }
    }
}

impl ChalkToRustc<rustc_ast_ir::Mutability> for chalk_ir::Mutability {
    fn to_rustc(&self) -> rustc_ast_ir::Mutability {
        match self {
            chalk_ir::Mutability::Mut => rustc_ast_ir::Mutability::Mut,
            chalk_ir::Mutability::Not => rustc_ast_ir::Mutability::Not,
        }
    }
}
