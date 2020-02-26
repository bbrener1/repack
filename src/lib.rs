#[macro_use]
extern crate ndarray;
extern crate num_traits;
extern crate ndarray_linalg;


use std::fmt::Debug;
use std::ops::{DivAssign,MulAssign,SubAssign,Mul,Neg};
use std::cmp::{max,min,PartialOrd,Ordering};
use ndarray::{Axis,Array,ArrayBase,ArrayViewMut2,ArrayViewMut1,ArrayView2,Array1,Array2,Ix2};
use num_traits::{Zero,One,Signed};
// use ndarray_linalg::layout::MatrixLayout;
use ndarray_linalg::lapack::{Pivot,Transpose,UPLO,into_result};
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

fn rgetrf<T:Zero + One + PartialOrd + Debug + LinalgScalar + ScalarOperand + DivAssign>(mut x:ArrayViewMut2<T>,mut ipiv: &mut [usize]) -> Option<()>
    where
        T: Signed + Zero + One + PartialOrd + Debug + LinalgScalar + ScalarOperand + DivAssign + MulAssign + SubAssign + Neg<Output=T>,
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

    // At this point we recurse again by calling rgetrf on A22.

    // TODO: We should probably consider implementing the block version as well.
    // TODO: Make sure boudns checks are all there.

    let (m,n) = x.dim();
    let k = min(m,n);

    if k == 0 {
        return Some(())
    }

    if m == 1 {
        if x[[0,0]] == T::zero() {return None}
        ipiv[0] = 0;
        return Some(())
    }
    else if n == 1 {
        // WE NEED SAFE MACHINE MINIMUM HERE
        // Write macro to compute

        // This is the single-column case. In the single column case we find the max, then swap it to the head of the column
        // After that we scale by the max

        let max = argmax(x.iter().map(|v| v.abs())).expect(&format!("Failed argmax: {:?}",x));
        x.swap([0,0],[max,0]);
        let max_val = x[[0,0]];
        if max_val != T::zero() {
            for i in 1..m {
                x[[i,0]] /= max_val;
            }
        }
        else {return None};
        ipiv[0] = max;

        // println!("Single");
        // println!("{:?}",x);
        // println!("{:?}",ipiv);
        // println!("==========");

        return Some(())

    }
    else {
        // This is the recursive case

        let n1 = k/2;

        // Here we make the recursive call to solve [A11]/[A21]
        //

        // println!("A11/A21");
        rgetrf(x.slice_mut(s![..,..n1]),&mut ipiv[..n1])?;

        // We perform the same swaps on [A12]/[A22] for synchronicity

        for (row,pivot) in ipiv[..n1].iter().enumerate() {
            row_swap(&mut x.slice_mut(s![..,n1..]), row, *pivot);
            // for j in n1..n {
            //     x.swap([row,j],[*pivot,j]);
            // }
        }

        // Now we solve for A12.

        {
            // This expression is enclosed in a block to allow the a11 and a12 references to drop
            // after execution is finished
            let (a11,a12) = x.slice_mut(s![..n1,..]).split_at(Axis(1),n1);
            rtrsm(a11.view(),a12,true,UPLO::Lower);
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

        // println!("A22");
        rgetrf(x.slice_mut(s![n1..,n1..]),&mut ipiv[n1..])?;

        // We update the pivot indices with the offset

        for i in ipiv[n1..].iter_mut() {
            *i += n1;
        }

        // Finally we perform the second set of pivots, which are updating A21

        for (row,pivot) in (n1..n).zip(ipiv[n1..].iter()) {
            row_swap(&mut x.slice_mut(s![..,..n1]), row, *pivot);
            // for j in 0..n1 {
            //     x.swap([row,j],[*pivot,j]);
            // }
        }

        // println!("{:?}",x);
        // println!("{:?}",ipiv);

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

    // Pretty sure this isn't robust yet. Compare against LAPACK reference later

    // TODO: bounds checks, quick error return

    let (m,n) = a.dim();

    for i in 0..(n-1) {
        let scale: T = b[i].neg();
        b.slice_mut(s![i+1..]).scaled_add(scale,&a.slice(s![i+1..,i]))
    }

}


pub fn rtrsm<T>(a: ArrayView2<T>,mut b: ArrayViewMut2<T>,unit:bool,uplo:UPLO)
where
    T: Zero + One + PartialOrd + Debug + LinalgScalar + ScalarOperand + DivAssign + MulAssign + SubAssign + Neg<Output=T>,
{
    // This is a more general triangular solver adapted from LAPACK 3.9.0 gtrsm
    // Operates only on values resting in the appropriate triangle

    // TODO bounds checks:
    //  0 dims

    let (m,n) = a.dim();

    match uplo {
        UPLO::Lower => {
            for j in 0..n {
                for k in 0..m {
                    let bkj = if unit {b[[k,j]]} else {b[[k,j]] / a[[k,k]]};
                    if bkj != T::zero() {
                        for i in k+1..m {
                            b[[i,j]] -= bkj * a[[i,k]];
                        }
                    }
                }
            }
        },
        UPLO::Upper => {
            for j in 0..n {
                for k in (0..m).rev() {
                    let bkj = if unit {b[[k,j]]} else {b[[k,j]] / a[[k,k]]};
                    if bkj != T::zero() {
                        for i in k+1..m {
                            b[[i,j]] -= bkj * a[[i,k]];
                        }
                    }
                }
            }
        },
    }


}

pub fn rgetrs<T>(a:ArrayView2<T>,mut b:ArrayViewMut2<T>,ipiv:&[usize])
where
    T: Signed + Zero + One + PartialOrd + Debug + LinalgScalar + ScalarOperand + DivAssign + MulAssign + SubAssign + Neg<Output=T>,
{
    // This is a general linear system solver that is used on the output of the LU decomposition.

    // Permutes B to a matching form for synchronicity
    for (row,pivot) in ipiv.iter().enumerate() {
        row_swap(&mut b, row, *pivot)
    }

    // Uses pre-written triangular solver

    rtrsm(a,b.slice_mut(s![..,..]),true,UPLO::Lower);
    rtrsm(a,b,false,UPLO::Upper);

}

pub fn rgecon<T>(a: ArrayView2<T>,norm:char,anorm:f64) {
    // Estimates the reciprocal of the condition number of a matrix given LU factorization

    //

    // Requires 1 & inf norm estimation of matrix and the inverse.

    //Layout:

    // 1. estimate 1 norm iteratively: dlacn2 port (holy shit dlacn2 is uhhh... not the best)
    // if failure: multiply by inv(L)
    //             multiply by inv(U)
    // or transpose depending on norm

    // scale by sl/su (a lot of weird conditionals)

}

pub fn rluinv<T>(mut a:ArrayViewMut2<T>,) -> Array2<T>
where
    T: Signed + Zero + One + PartialOrd + Debug + LinalgScalar + ScalarOperand + DivAssign + MulAssign + SubAssign + Neg<Output=T>,
{
    // Inverts a matrix in place via LU decomposition
    // Very hacked out, for the moment
    // More bounds checks?
    // Model on existing LAPACK?

    let (m,n) = a.dim();
    if m!=n {panic!("Tried to invert an array that wasn't square")}
    let k = m;
    let mut ipiv = vec![0;k];
    rgetrf(a.view_mut(),&mut ipiv).expect("Matrix is singular");
    let mut id: Array2<T> = Array::eye(k);
    rgetrs(a.view(),id.view_mut(),&ipiv);
    return id
}

pub fn row_swap<T>(a:&mut ArrayViewMut2<T>,r1:usize,r2:usize) {
    for i in 0..a.dim().1 {
        a.swap([r1,i],[r2,i]);
    }
}


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

    fn generic_square_f_2() -> Array2<f64> {
        array![
            [  4.,  12., -16.],
            [ 12.,  37., -43.],
            [-16., -43.,  98.]
        ]
    }

    fn generic_square_f_3() -> Array2<f64> {
        array![[4.,2.,3.],
               [1.,5.,-2.],
               [1.,1.,-3.]]
    }

    fn generic_square_f_4() -> Array2<f64> {
        array![[4.,2.,3.],
               [2.,5.,-2.],
               [3.,-2.,-3.]]
    }

    fn generic_square_f_5() -> Array2<f64> {
        array![ [ -2.,   8., -10.],
                [  4.,  -2.,   5.],
                [ -8.,   8.,  -6.]]
    }

    fn singular_square_f_6() -> Array2<f64> {
        array![ [ -2.,   8., -10., 0.],
                [  4.,  -2.,   5., 1.],
                [ -8.,   8.,  -6., 3.],
                [ -4.,   16., -20., 0.]]
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
        rtrsm(m.view(),b.view_mut(),true,UPLO::Lower);
        println!("{:?}",b);
        println!("{:?}",m.dot(&b));
        panic!();
    }

    #[test]
    fn rgetrf_test() {
        let mut square = generic_square_f();
        let mut pivots = vec![0;3];
        rgetrf(square.view_mut(),&mut pivots);
        println!("{:?}",square);
        println!("{:?}",pivots);
        panic!();
    }

    #[test]
    fn rgetrf_test_2() {
        let mut square = generic_square_f_2();
        let mut pivots = vec![0;3];
        rgetrf(square.view_mut(),&mut pivots);
        println!("{:?}",square);
        println!("{:?}",pivots);
        panic!();
    }

    #[test]
    fn rgetrf_test_3() {
        let mut square = generic_square_f_3();
        let mut pivots = vec![0;3];
        rgetrf(square.view_mut(),&mut pivots);
        println!("{:?}",square);
        println!("{:?}",pivots);
        panic!();
    }

    #[test]
    fn rgetrf_test_4() {
        let mut square = generic_square_f_4();
        let mut pivots = vec![0;3];
        rgetrf(square.view_mut(),&mut pivots);
        println!("{:?}",square);
        println!("{:?}",pivots);
        panic!();
    }

    #[test]
    fn rgetrf_test_5() {
        let mut square = generic_square_f_5();
        let mut pivots = vec![0;3];
        rgetrf(square.view_mut(),&mut pivots);
        println!("{:?}",square);
        println!("{:?}",pivots);
        panic!();
    }

    #[test]
    fn rgetrf_test_singular() {
        let mut square = singular_square_f_6();
        let mut pivots = vec![0;4];
        rgetrf(square.view_mut(),&mut pivots);
        println!("{:?}",square);
        println!("{:?}",pivots);
        panic!();
    }

    #[test]
    fn lu_lapack_test() {

        use ndarray_linalg::solve::Factorize;

        let mut square = generic_square_f();
        let mut pivots = vec![0;3];
        let f = square.factorize().unwrap();
        println!("{:?}",square);
        println!("{:?}",f.a);
        panic!();
    }

    #[test]
    fn lu_lapack_singular_test() {

        use ndarray_linalg::solve::Factorize;

        let mut square = singular_square_f_6();
        let mut pivots = vec![0;4];
        let f = square.factorize().unwrap();
        println!("{:?}",square);
        println!("{:?}",f.a);
        panic!();
    }

}
