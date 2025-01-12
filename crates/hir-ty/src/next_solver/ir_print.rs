use std::any::type_name_of_val;

use rustc_type_ir::{self as ty, ir_print::IrPrint};

use super::interner::DbInterner;

impl IrPrint<ty::AliasTy<Self>> for DbInterner {
    fn print(t: &ty::AliasTy<Self>, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        crate::next_solver::tls::with_opt_db_out_of_thin_air(|db| {
            match db {
                Some(db) => {
                    let alias_ = match t.def_id {
                        hir_def::GenericDefId::TypeAliasId(id) => id,
                        _ => panic!("Expected TypeAlais."),
                    };
                    fmt.write_str(&format!("AliasTy({:?}[{:?}])", db.type_alias_data(alias_).name.as_str(), t.args))
                }
                None => fmt.write_str(&format!("AliasTy({:?}[{:?}])", t.def_id, t.args)),
            }
        })
    }

    fn print_debug(t: &ty::AliasTy<Self>, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        crate::next_solver::tls::with_opt_db_out_of_thin_air(|db| {
            match db {
                Some(db) => {
                    let alias_ = match t.def_id {
                        hir_def::GenericDefId::TypeAliasId(id) => id,
                        _ => panic!("Expected TypeAlais."),
                    };
                    fmt.write_str(&format!("AliasTy({:?}[{:?}])", db.type_alias_data(alias_).name.as_str(), t.args))
                }
                None => fmt.write_str(&format!("AliasTy({:?}[{:?}])", t.def_id, t.args)),
            }
        })
    }
}

impl IrPrint<ty::AliasTerm<Self>> for DbInterner {
    fn print(t: &ty::AliasTerm<Self>, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        crate::next_solver::tls::with_opt_db_out_of_thin_air(|db| {
            match db {
                Some(db) => {
                    let alias_ = match t.def_id {
                        hir_def::GenericDefId::TypeAliasId(id) => id,
                        _ => panic!("Expected TypeAlais."),
                    };
                    fmt.write_str(&format!("AliasTerm({:?}[{:?}])", db.type_alias_data(alias_).name.as_str(), t.args))
                }
                None => fmt.write_str(&format!("AliasTerm({:?}[{:?}])", t.def_id, t.args)),
            }
        })
    }

    fn print_debug(t: &ty::AliasTerm<Self>, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        crate::next_solver::tls::with_opt_db_out_of_thin_air(|db| {
            match db {
                Some(db) => {
                    let alias_ = match t.def_id {
                        hir_def::GenericDefId::TypeAliasId(id) => id,
                        _ => panic!("Expected TypeAlais."),
                    };
                    fmt.write_str(&format!("AliasTerm({:?}[{:?}])", db.type_alias_data(alias_).name.as_str(), t.args))
                }
                None => fmt.write_str(&format!("AliasTerm({:?}[{:?}])", t.def_id, t.args)),
            }
        })
    }
}
impl IrPrint<ty::TraitRef<Self>> for DbInterner {
    fn print(t: &ty::TraitRef<Self>, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        crate::next_solver::tls::with_opt_db_out_of_thin_air(|db| {
            match db {
                Some(db) => {
                    let trait_ = match t.def_id {
                        hir_def::GenericDefId::TraitId(id) => id,
                        _ => panic!("Expected trait."),
                    };
                    fmt.write_str(&format!("TraitRef({:?}[{:?}])", db.trait_data(trait_).name.as_str(), t.args))
                }
                None => fmt.write_str(&format!("TraitRef({:?}[{:?}])", t.def_id, t.args)),
            }
        })
    }

    fn print_debug(t: &ty::TraitRef<Self>, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        crate::next_solver::tls::with_opt_db_out_of_thin_air(|db| {
            match db {
                Some(db) => {
                    let trait_ = match t.def_id {
                        hir_def::GenericDefId::TraitId(id) => id,
                        _ => panic!("Expected trait."),
                    };
                    fmt.write_str(&format!("TraitRef({:?}[{:?}])", db.trait_data(trait_).name.as_str(), t.args))
                }
                None => fmt.write_str(&format!("TraitRef({:?}[{:?}])", t.def_id, t.args)),
            }
        })
    }
}
impl IrPrint<ty::TraitPredicate<Self>> for DbInterner {
    fn print(t: &ty::TraitPredicate<Self>, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.write_str(&format!("TODO: {:?}", type_name_of_val(t)))
    }

    fn print_debug(
        t: &ty::TraitPredicate<Self>,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        fmt.write_str(&format!("TODO: {:?}", type_name_of_val(t)))
    }
}
impl IrPrint<rustc_type_ir::HostEffectPredicate<Self>> for DbInterner {
    fn print(
        t: &rustc_type_ir::HostEffectPredicate<Self>,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        fmt.write_str(&format!("TODO: {:?}", type_name_of_val(t)))
    }

    fn print_debug(
        t: &rustc_type_ir::HostEffectPredicate<Self>,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        fmt.write_str(&format!("TODO: {:?}", type_name_of_val(t)))
    }
}
impl IrPrint<ty::ExistentialTraitRef<Self>> for DbInterner {
    fn print(
        t: &ty::ExistentialTraitRef<Self>,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        crate::next_solver::tls::with_opt_db_out_of_thin_air(|db| {
            match db {
                Some(db) => {
                    let trait_ = match t.def_id {
                        hir_def::GenericDefId::TraitId(id) => id,
                        _ => panic!("Expected trait."),
                    };
                    fmt.write_str(&format!("ExistentialTraitRef({:?}[{:?}])", db.trait_data(trait_).name.as_str(), t.args))
                }
                None => fmt.write_str(&format!("ExistentialTraitRef({:?}[{:?}])", t.def_id, t.args)),
            }
        })
    }

    fn print_debug(
        t: &ty::ExistentialTraitRef<Self>,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        crate::next_solver::tls::with_opt_db_out_of_thin_air(|db| {
            match db {
                Some(db) => {
                    let trait_ = match t.def_id {
                        hir_def::GenericDefId::TraitId(id) => id,
                        _ => panic!("Expected trait."),
                    };
                    fmt.write_str(&format!("ExistentialTraitRef({:?}[{:?}])", db.trait_data(trait_).name.as_str(), t.args))
                }
                None => fmt.write_str(&format!("ExistentialTraitRef({:?}[{:?}])", t.def_id, t.args)),
            }
        })
    }
}
impl IrPrint<ty::ExistentialProjection<Self>> for DbInterner {
    fn print(
        t: &ty::ExistentialProjection<Self>,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        crate::next_solver::tls::with_opt_db_out_of_thin_air(|db| {
            match db {
                Some(db) => {
                    let id = match t.def_id {
                        hir_def::GenericDefId::TypeAliasId(id) => id,
                        _ => panic!("Expected trait."),
                    };
                    fmt.write_str(&format!("ExistentialProjection(({:?}[{:?}]) -> {:?})", db.type_alias_data(id).name.as_str(), t.args, t.term))
                }
                None => fmt.write_str(&format!("ExistentialProjection(({:?}[{:?}]) -> {:?})", t.def_id, t.args, t.term)),
            }
        })
    }

    fn print_debug(
        t: &ty::ExistentialProjection<Self>,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        crate::next_solver::tls::with_opt_db_out_of_thin_air(|db| {
            match db {
                Some(db) => {
                    let id = match t.def_id {
                        hir_def::GenericDefId::TypeAliasId(id) => id,
                        _ => panic!("Expected trait."),
                    };
                    fmt.write_str(&format!("ExistentialProjection(({:?}[{:?}]) -> {:?})", db.type_alias_data(id).name.as_str(), t.args, t.term))
                }
                None => fmt.write_str(&format!("ExistentialProjection(({:?}[{:?}]) -> {:?})", t.def_id, t.args, t.term)),
            }
        })
    }
}
impl IrPrint<ty::ProjectionPredicate<Self>> for DbInterner {
    fn print(
        t: &ty::ProjectionPredicate<Self>,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        fmt.write_str(&format!("TODO: {:?}", type_name_of_val(t)))
    }

    fn print_debug(
        t: &ty::ProjectionPredicate<Self>,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        fmt.write_str(&format!("TODO: {:?}", type_name_of_val(t)))
    }
}
impl IrPrint<ty::NormalizesTo<Self>> for DbInterner {
    fn print(t: &ty::NormalizesTo<Self>, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.write_str(&format!("TODO: {:?}", type_name_of_val(t)))
    }

    fn print_debug(
        t: &ty::NormalizesTo<Self>,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        fmt.write_str(&format!("TODO: {:?}", type_name_of_val(t)))
    }
}
impl IrPrint<ty::SubtypePredicate<Self>> for DbInterner {
    fn print(
        t: &ty::SubtypePredicate<Self>,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        fmt.write_str(&format!("TODO: {:?}", type_name_of_val(t)))
    }

    fn print_debug(
        t: &ty::SubtypePredicate<Self>,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        fmt.write_str(&format!("TODO: {:?}", type_name_of_val(t)))
    }
}
impl IrPrint<ty::CoercePredicate<Self>> for DbInterner {
    fn print(t: &ty::CoercePredicate<Self>, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.write_str(&format!("TODO: {:?}", type_name_of_val(t)))
    }

    fn print_debug(
        t: &ty::CoercePredicate<Self>,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        fmt.write_str(&format!("TODO: {:?}", type_name_of_val(t)))
    }
}
impl IrPrint<ty::FnSig<Self>> for DbInterner {
    fn print(t: &ty::FnSig<Self>, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.write_str(&format!("TODO: {:?}", type_name_of_val(t)))
    }

    fn print_debug(t: &ty::FnSig<Self>, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.write_str(&format!("TODO: {:?}", type_name_of_val(t)))
    }
}
