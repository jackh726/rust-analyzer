use extension_traits::extension;
use rustc_abi::{Float, HasDataLayout, Integer, IntegerType, Primitive, ReprOptions};
use rustc_type_ir::{CoroutineArgs, FloatTy, IntTy, UintTy};

use super::{DbInterner, Ty, TyKind};

#[derive(Clone, Debug)]
pub struct Discr {
    /// Bit representation of the discriminant (e.g., `-128i8` is `0xFF_u128`).
    pub val: u128,
    pub ty: Ty,
}

impl Discr {
    /// Adds `1` to the value and wraps around if the maximum for the type is reached.
    pub fn wrap_incr(self, interner: DbInterner) -> Self {
        self.checked_add(interner, 1).0
    }
    pub fn checked_add(self, interner: DbInterner, n: u128) -> (Self, bool) {
        let (size, signed) = self.ty.clone().int_size_and_signed(interner);
        let (val, oflo) = if signed {
            let min = size.signed_int_min();
            let max = size.signed_int_max();
            let val = size.sign_extend(self.val);
            assert!(n < (i128::MAX as u128));
            let n = n as i128;
            let oflo = val > max - n;
            let val = if oflo { min + (n - (max - val) - 1) } else { val + n };
            // zero the upper bits
            let val = val as u128;
            let val = size.truncate(val);
            (val, oflo)
        } else {
            let max = size.unsigned_int_max();
            let val = self.val;
            let oflo = val > max - n;
            let val = if oflo { n - (max - val) - 1 } else { val + n };
            (val, oflo)
        };
        (Self { val, ty: self.ty }, oflo)
    }
}

#[extension(pub trait IntegerTypeExt)]
impl IntegerType {
    fn to_ty(&self, interner: DbInterner) -> Ty {
        match self {
            IntegerType::Pointer(true) => Ty::new(TyKind::Int(IntTy::Isize)),
            IntegerType::Pointer(false) => Ty::new(TyKind::Uint(UintTy::Usize)),
            IntegerType::Fixed(i, s) => i.to_ty(interner, *s),
        }
    }

    fn initial_discriminant(&self, interner: DbInterner) -> Discr {
        Discr { val: 0, ty: self.to_ty(interner) }
    }

    fn disr_incr(&self, interner: DbInterner, val: Option<Discr>) -> Option<Discr> {
        if let Some(val) = val {
            assert_eq!(self.to_ty(interner), val.ty);
            let (new, oflo) = val.checked_add(interner, 1);
            if oflo {
                None
            } else {
                Some(new)
            }
        } else {
            Some(self.initial_discriminant(interner))
        }
    }
}

#[extension(pub trait IntegerExt)]
impl Integer {
    #[inline]
    fn to_ty(&self, interner: DbInterner, signed: bool) -> Ty {
        use Integer::*;
        match (*self, signed) {
            (I8, false) => Ty::new(TyKind::Uint(UintTy::U8)),
            (I16, false) => Ty::new(TyKind::Uint(UintTy::U16)),
            (I32, false) => Ty::new(TyKind::Uint(UintTy::U32)),
            (I64, false) => Ty::new(TyKind::Uint(UintTy::U64)),
            (I128, false) => Ty::new(TyKind::Uint(UintTy::U128)),
            (I8, true) => Ty::new(TyKind::Int(IntTy::I8)),
            (I16, true) => Ty::new(TyKind::Int(IntTy::I16)),
            (I32, true) => Ty::new(TyKind::Int(IntTy::I32)),
            (I64, true) => Ty::new(TyKind::Int(IntTy::I64)),
            (I128, true) => Ty::new(TyKind::Int(IntTy::I128)),
        }
    }

    fn from_int_ty<C: HasDataLayout>(cx: &C, ity: IntTy) -> Integer {
        use Integer::*;
        match ity {
            IntTy::I8 => I8,
            IntTy::I16 => I16,
            IntTy::I32 => I32,
            IntTy::I64 => I64,
            IntTy::I128 => I128,
            IntTy::Isize => cx.data_layout().ptr_sized_integer(),
        }
    }
    fn from_uint_ty<C: HasDataLayout>(cx: &C, ity: UintTy) -> Integer {
        use Integer::*;
        match ity {
            UintTy::U8 => I8,
            UintTy::U16 => I16,
            UintTy::U32 => I32,
            UintTy::U64 => I64,
            UintTy::U128 => I128,
            UintTy::Usize => cx.data_layout().ptr_sized_integer(),
        }
    }

    /// Finds the appropriate Integer type and signedness for the given
    /// signed discriminant range and `#[repr]` attribute.
    /// N.B.: `u128` values above `i128::MAX` will be treated as signed, but
    /// that shouldn't affect anything, other than maybe debuginfo.
    fn repr_discr(
        interner: DbInterner,
        ty: Ty,
        repr: &ReprOptions,
        min: i128,
        max: i128,
    ) -> (Integer, bool) {
        // Theoretically, negative values could be larger in unsigned representation
        // than the unsigned representation of the signed minimum. However, if there
        // are any negative values, the only valid unsigned representation is u128
        // which can fit all i128 values, so the result remains unaffected.
        let unsigned_fit = Integer::fit_unsigned(std::cmp::max(min as u128, max as u128));
        let signed_fit = std::cmp::max(Integer::fit_signed(min), Integer::fit_signed(max));

        if let Some(ity) = repr.int {
            let discr = Integer::from_attr(&interner, ity);
            let fit = if ity.is_signed() { signed_fit } else { unsigned_fit };
            if discr < fit {
                panic!(
                    "Integer::repr_discr: `#[repr]` hint too small for \
                      discriminant range of enum `{:?}`",
                    ty
                )
            }
            return (discr, ity.is_signed());
        }

        let at_least = if repr.c() {
            // This is usually I32, however it can be different on some platforms,
            // notably hexagon and arm-none/thumb-none
            interner.data_layout().c_enum_min_size
        } else {
            // repr(Rust) enums try to be as small as possible
            Integer::I8
        };

        // If there are no negative values, we can use the unsigned fit.
        if min >= 0 {
            (std::cmp::max(unsigned_fit, at_least), false)
        } else {
            (std::cmp::max(signed_fit, at_least), true)
        }
    }
}

#[extension(pub trait FloatExt)]
impl Float {
    #[inline]
    fn to_ty(&self, interner: DbInterner) -> Ty {
        use Float::*;
        match *self {
            F16 => Ty::new(TyKind::Float(FloatTy::F16)),
            F32 => Ty::new(TyKind::Float(FloatTy::F32)),
            F64 => Ty::new(TyKind::Float(FloatTy::F64)),
            F128 => Ty::new(TyKind::Float(FloatTy::F128)),
        }
    }

    fn from_float_ty(fty: FloatTy) -> Self {
        use Float::*;
        match fty {
            FloatTy::F16 => F16,
            FloatTy::F32 => F32,
            FloatTy::F64 => F64,
            FloatTy::F128 => F128,
        }
    }
}

#[extension(pub trait PrimitiveExt)]
impl Primitive {
    #[inline]
    fn to_ty(&self, interner: DbInterner) -> Ty {
        match *self {
            Primitive::Int(i, signed) => i.to_ty(interner, signed),
            Primitive::Float(f) => f.to_ty(interner),
            Primitive::Pointer(_) => Ty::new(TyKind::RawPtr(
                Ty::new(TyKind::Tuple(Default::default())),
                rustc_ast_ir::Mutability::Mut,
            )),
        }
    }

    /// Return an *integer* type matching this primitive.
    /// Useful in particular when dealing with enum discriminants.
    #[inline]
    fn to_int_ty(&self, interner: DbInterner) -> Ty {
        match *self {
            Primitive::Int(i, signed) => i.to_ty(interner, signed),
            Primitive::Pointer(_) => {
                let signed = false;
                interner.data_layout().ptr_sized_integer().to_ty(interner, signed)
            }
            Primitive::Float(_) => panic!("floats do not have an int type"),
        }
    }
}

impl HasDataLayout for DbInterner {
    fn data_layout(&self) -> &rustc_abi::TargetDataLayout {
        todo!()
    }
}

#[extension(pub trait CoroutineArgsExt)]
impl CoroutineArgs<DbInterner> {
    /// The type of the state discriminant used in the coroutine type.
    #[inline]
    fn discr_ty(&self, interner: DbInterner) -> Ty {
        Ty::new(TyKind::Uint(UintTy::U32))
    }
}
