#[macro_use]
extern crate ndarray;
extern crate num_traits;
// extern crate ndarray_linalg;


use std::fmt::Debug;
use std::ops::{DivAssign,MulAssign,SubAssign,Mul,Neg};
use std::cmp::{max,min,PartialOrd,Ordering};
use ndarray::{Axis,ArrayBase,ArrayViewMut2,ArrayViewMut1,ArrayView2,Array1,Array2,Ix2};
use num_traits::{Zero,One,CheckedNeg};
// use ndarray_linalg::layout::MatrixLayout;
// use ndarray_linalg::lapack::{Pivot,Transpose,UPLO,into_result};
// use ndarray_linalg::error::*;
// use ndarray_linalg::triangular::{Diag};
// use ndarray_linalg::{Scalar};
use ndarray::{LinalgScalar,ScalarOperand};
use std::mem::swap;

const SAFE_MIN:f64 = 1e-200;

// pub trait SolvePure_: Scalar + Sized {
//     fn lu(l: MatrixLayout, a: &mut [Self]) -> Result<Pivot>;
//     fn inv(l: MatrixLayout, a: &mut [Self], p: &Pivot) -> Result<()>;
//
//     // fn rcond(l: MatrixLayout, a: &[Self], anorm: Self::Real) -> Result<Self::Real>;
//     // fn solve(l: MatrixLayout, t: Transpose, a: &[Self], p: &Pivot, b: &mut [Self]) -> Result<()>;
// }

// impl SolvePure_ for f64 {
//     fn lu(l: MatrixLayout, a: &mut [Self]) -> Result<Pivot> {
//         let (m, n) = l.size();
//         let k = min(m,n);
//         let mut pivots = Pivot::with_capacity(k as usize);
//
//         into_result(-1, vec![])
//     }
//
//     fn inv(l: MatrixLayout, a: &mut [Self], ipiv: &Pivot) -> Result<()> { into_result(-1, ()) }
// }
//
//

fn rgetrf2<T:Zero + One + PartialOrd + Debug + LinalgScalar + ScalarOperand + DivAssign>(mut x:ArrayViewMut2<T>,mut ipiv: &mut [usize]) -> Option<()>
    where
        T: Zero + One + PartialOrd + Debug + LinalgScalar + ScalarOperand + DivAssign + MulAssign + SubAssign + Neg<Output=T>,
{

    // Port of the LAPACK 3.9.0 dgetrf2 LU Factorization Algorithm
    // This is a recursive procedure with pivoting and row interchanges (whatever that means)

    // It produces a unit lower triangular, and general upper triangular matrix, as well as P, a permutation matrix

    // The layout of the algorithm relies on recursive calling (without reallocation) of the solver on quadrants of a matrix

    // Shamelessly copy-pasted from the LAPACK documentation:

    //      [  A11 | A12  ]  where A11 is n1 by n1 and A22 is n2 by n2
    //  A = [ -----|----- ]  with n1 = min(m,n)/2
    //      [  A21 | A22  ]       n2 = n-n1

    // Recursion occurs on A11/A21, after which you get a set of pivots. The pivots are then re-applied to A12/A22 in order to keep the matrix
    // synchronous.

    // At this point A11 is lower-triangular. Together with A12 it forms an equation A1(X)=A12.

    // We replace A12 with X by calling rtrsm on A11,A12.

    // Then we "update" A22 by calling a matrix multiplication A22 -= A21 * A12.

    // At this point we recurse again by calling rgetrf2 on A22.

    // ON THE BUFFER:

    // A fundamental limitation of the ndarray rust implementation is that you cannot borrow multiple mutable slices of the same array,
    // even if they do not overlap. For this reason we will need a buffer in which to place the outputs of operatatons involving parts of the array


    //

    let (m,n) = x.dim();
    let k = min(m,n);

    if k == 0 {
        return Some(())
    }

    if m == 1 {
        x[[0,0]] = T::one();
        ipiv[0] = 0;
        return Some(())
    }
    else if n == 1 {
        // WE NEED SAFE MACHINE MINIMUM HERE
        // Write macro to compute

        // This is the single-column case. In the single column case we find the max, then swap it to the head of the column
        // After that we scale by the max

        let max = argmax(x.iter()).expect(&format!("Failed argmax: {:?}",x));
        x.swap([0,0],[max,0]);
        let max_val = x[[0,0]];
        x /= max_val;

        ipiv[0] = max;


    }
    else {
        // This is the recursive case

        let n1 = k/2;
        let n2 = n-n1;

        // Here we make the recursive call to solve [A11]/[A21]
        //
        rgetrf2(x.slice_mut(s![..,..n1]),&mut ipiv[..n1]);

        // We perform the same swaps on [A12]/[A22] for synchronicity

        for (row,pivot) in ipiv[..n1].iter().enumerate() {
            for j in n1..n {
                x.swap([row,j],[*pivot,j]);
            }
        }

        // Now we solve for A12.

        {
            // This expression is enclosed in a block to allow the a11 and a12 references to drop
            // after execution is finished
            let (a11,a12) = x.slice_mut(s![..n1,..]).split_at(Axis(1),n1);
            rtrsm(a11.view(),a12);
        }

        // We update A22:

        {
            let (upper_view,lower_view) = x.view_mut().split_at(Axis(0),n1);
            let a12 = upper_view.slice(s![..,n1..]);
            let (a21,mut a22) = lower_view.split_at(Axis(1),n1);

            // In order to avoid large allocations here I am going to try a manual multiplication.

            for i in 0..a21.dim().0 {
                for j in 0..a12.dim().1 {
                    a22[[i,j]] -= a21.row(i).dot(&(a12.column(j)));
                }
            }

            // An alternative:
            // a22 -= &a12.dot(&a22);
        }

        // We recursively factor A22

        rgetrf2(x.slice_mut(s![n1..,n1..]),&mut ipiv[n1..]);

        // We update the pivot indices with the offset

        for i in ipiv[n1..].iter_mut() {
            *i += n1;
        }

        // Finally we perform the second set of pivots, which are updating A21

        for (row,pivot) in (n1..n).zip(ipiv[n1..].iter()) {
            for j in 0..n1 {
                x.swap([row,j],[*pivot,j]);
            }
        }

        return Some(())

    }

    None

}


pub fn forward_substitution<T>(a: ArrayView2<T>,mut b: ArrayViewMut1<T>)
where
    T: Zero + One + PartialOrd + Debug + LinalgScalar + ScalarOperand + DivAssign + MulAssign + SubAssign + Neg<Output=T>,
{
    // This is a forward substitution procedure solving A * X = B, assuming A is a lower unit triangular
    // B is replaced by a solution

    // TODO: bounds checks, quick error return

    let (m,n) = a.dim();

    for i in 0..(n-1) {
        let scale: T = b[i].neg();
        b.slice_mut(s![i+1..]).scaled_add(scale,&a.slice(s![i+1..,i]))
    }

}


pub fn rtrsm<T>(a: ArrayView2<T>,mut b: ArrayViewMut2<T>)
where
    T: Zero + One + PartialOrd + Debug + LinalgScalar + ScalarOperand + DivAssign + MulAssign + SubAssign + Neg<Output=T>,
{
    // This is a more general triangular solver adapted from LAPACK

    // TODO bounds checks:
    //  0 dims

    //TODO scaling?


    let (m,n) = a.dim();

    for j in 0..n {
        for k in 0..m {
            let bkj = b[[k,j]];
            if bkj != T::zero() {
                for i in k+1..m {
                    b[[i,j]] -= bkj * a[[i,k]];
                }
            }
        }
    }
}

//
// pub fn rtrsm_unified<T>(mut a: ArrayViewMut2<T>,m:usize,n:usize)
// where
//     T:  Zero + One + PartialOrd + Debug + LinalgScalar + ScalarOperand + DivAssign + MulAssign + SubAssign + Neg<Output=T>,
// {
//     // ndarray-rust cannot borrow two mutable subslices from the same array, so if we want to do this with
//     // few allocations we have to do some gymnastics.
//
//     // This function solves an arbitrary triangular matrix equation A * X = B where A is unit lower triangular.
//     // Unlike the general rtrsm, both A and B are contained in the same mutable array view, stacked by column
//
//     // a is the [A | B] block matrix. The B block is replaced by the solution X.
//
//     // For this reason the argument must specify the dimension of A. (Dimension of B is implied)
//
//     let (um, un) = a.dim();
//
//     // We compute the offsets for indexing into B:
//
//     let (bm,bn) = (0,un-n);
//
//     for j in 0..n {
//         for k in 0..m {
//             let bkj = a[[bm+k,bn+j]];
//             if bkj != T::zero() {
//                 for i in k+1..m {
//                     let v = a[[i,k]];
//                     a[[bm+i,bn+j]] -= bkj * v;
//                 }
//             }
//         }
//     }
// }

// fn solve_triangular(&self, uplo: UPLO, diag: Diag, b: &ArrayBase<S, D>) -> Result<Array<A, D>>;

pub fn argmax<T:Iterator<Item=U>,U:PartialOrd + PartialEq>(input: T) -> Option<usize> {
    let mut maximum: Option<(usize,U)> = None;
    for (j,val) in input.enumerate() {
        let check = if let Some((i,m)) = maximum.take() {
            let ordering = val.partial_cmp(&m).expect("Nan in argmax");
            match ordering {
                Ordering::Less => {Some((i,m))},
                Ordering::Equal => {Some((i,m))},
                Ordering::Greater => {Some((j,val))},
            }
        }
        else {
            if val.partial_cmp(&val).is_some() { Some((j,val)) }
            else { None }
        };
        maximum = check;

    };
    maximum.map(|(i,m)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn argmax_test() {
    //     let mut v = vec![1f64,3f64,5f64,3f64];
    //     assert_eq!(argmax(v.iter()),Some(2));
    //     v[1] = "NaN".parse::<f64>().unwrap();
    //     assert_eq!(argmax(v.iter()),None);
    // }

    fn lower_triangular_f() -> (Array2<f64>,Array1<f64>) {
        (array![[1.,0.,0.],[2.,1.,0.],[1.,0.,1.]],array![4.,2.,3.])
    }

    fn lower_triangular_i() -> (Array2<i32>,Array1<i32>) {
        (array![[1,0,0],[2,1,0],[1,0,1]],array![4,2,3])
    }

    fn generic_square_f() -> Array2<f64> {
        array![[4.,2.,3.],
               [0.,5.,-2.],
               [0.,1.,-3.]]        
    }

    fn lower_triangular_f_square() -> (Array2<f64>,Array2<f64>) {
        (array![[1.,0.,0.],
                [2.,1.,0.],
                [1.,0.,1.]],

        array![[4.,2.,3.],
               [0.,5.,-2.],
               [0.,1.,-3.]])
    }

    fn lower_triangular_f_square_unified() -> Array2<f64> {
        array![[1.,0.,0.,4.,2.,3.],
                [2.,1.,0.,0.,5.,-2.],
                [1.,0.,1.,0.,1.,-3.]]
    }

    #[test]
    fn forward_substitution_test_f() {
        let (mut m,mut b) = lower_triangular_f();
        forward_substitution(m.view(),b.view_mut());
        println!("{:?}",b);
        println!("{:?}",m.dot(&b));
        panic!();
    }

    #[test]
    fn forward_substitution_test_i() {
        let (mut m,mut b) = lower_triangular_i();
        forward_substitution(m.view(),b.view_mut());
        println!("{:?}",b);
        println!("{:?}",m.dot(&b));
        panic!();
    }

    #[test]
    fn rtrsm_test_f() {
        let (mut m,mut b) = lower_triangular_f_square();
        rtrsm(m.view(),b.view_mut());
        println!("{:?}",b);
        println!("{:?}",m.dot(&b));
        panic!();
    }
    //
    // #[test]
    // fn rtrsm_test_f_unified() {
    //     let mut m = lower_triangular_f_square_unified();
    //     rtrsm_unified(m.view_mut(),3,3);
    //     println!("{:?}",m);
    //     let a = m.slice(s![..,..3]);
    //     let b = m.slice(s![..,3..]);
    //     println!("{:?}",a.dot(&b));
    //     panic!();
    // }


}
