"""
Homework4.
Replace 'pass' by your implementation.
"""
import numpy as np
import sympy
from helper import refineF, gaussian_filter_2d
from tqdm import tqdm
from scipy.optimize import leastsq
'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''


def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation

    # Step 0: normalize
    normalized_pts1 = np.array(pts1)/M
    normalized_pts2 = np.array(pts2)/M
    num_samples = normalized_pts1.shape[0]

    # Step 1: Construct A matrix
    A = []
    for i in range(num_samples):
        x1, y1 = list(normalized_pts1[i])
        x2, y2 = list(normalized_pts2[i])

        row = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]
        A.append(row)
    A = np.array(A)

    # Step 2: Use SVD to find F
    _, _, VT = np.linalg.svd(A)
    F = VT[-1].reshape(3, 3)

    # Step3: Enforce rank-2 of F
    U, Sigma, VT = np.linalg.svd(F)
    rank2_F = np.linalg.multi_dot([U[:, :2], np.diag(Sigma[:2]), VT[:2]])

    # Step4: Refine F
    refined_rank2_F = refineF(rank2_F, normalized_pts1, normalized_pts2)

    # Step5: Un-normalize F (if x'=Tx, then F_un-normalized = T^T(F)T)
    T = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]])
    F_unnormalized = np.linalg.multi_dot([T.T, refined_rank2_F, T])

    return F_unnormalized


'''
Q2.2: Seven Point Algorithm
    Input:  pts1, 7x2 Matrix
            pts2, 7x2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''


def sevenpoint(pts1, pts2, M):
    # Replace pass by your implementation

    # Step 0: normalize
    assert pts1.shape[0] == 7 and pts2.shape[0] == 7

    normalized_pts1 = np.array(pts1)/M
    normalized_pts2 = np.array(pts2)/M

    # Step 1: Construct A matrix
    A = []
    for i in range(normalized_pts2.shape[0]):
        x1, y1 = list(normalized_pts1[i])
        x2, y2 = list(normalized_pts2[i])

        row = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]
        A.append(row)
    A = np.array(A)

    # Step 2: Use SVD to find F, use the polynomial constraint
    _, _, VT = np.linalg.svd(A)
    F1 = VT[-2].reshape(3, 3)
    F2 = VT[-1].reshape(3, 3)

    sympy.var('a')
    F = sympy.Matrix(F1) + sympy.Matrix(F2)*a
    detF = sympy.det(F)
    coeff = np.array([detF.coeff(a**3), detF.coeff(a**1),
                     detF.coeff(a**2), detF.subs(a, 0)])
    lambdas = list(np.roots(coeff))
    Farray = []
    for l in lambdas:
        if l.imag == 0.0:
            _lambda = l.real
            F = (VT[-2] + VT[-1]*_lambda).reshape((3, 3))

            # Step4: Refine F
            refined_F = refineF(F, normalized_pts1, normalized_pts2)

            # Step5: Un-normalize F
            T = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]])
            F_unnormalized = np.linalg.multi_dot([T.T, refined_F, T])

            Farray.append(F_unnormalized)

    return Farray


def sevenpoint_v2(pts1, pts2, M):
    # Replace pass by your implementation
    pts1 = pts1 * 1.0 / M
    pts2 = pts2 * 1.0 / M
    n, temp = pts1.shape
    x1 = pts1[:, 0]
    y1 = pts1[:, 1]
    x2 = pts2[:, 0]
    y2 = pts2[:, 1]
    A1 = (x2 * x1)
    A2 = (x2 * y1)
    A3 = x2
    A4 = y2 * x1
    A5 = y2 * y1
    A6 = y2
    A7 = x1
    A8 = y1
    A9 = np.ones(n)
    A = np.vstack((A1, A2, A3, A4, A5, A6, A7, A8, A9))
    A = A.T
    u, s, vh = np.linalg.svd(A)
    f1 = vh[-1, :]
    f2 = vh[-2, :]
    F1 = f1.reshape(3, 3)
    F2 = f2.reshape(3, 3)
    F1 = refineF(F1, pts1, pts2)
    F2 = refineF(F2, pts1, pts2)

    # w = sympy.Symbol('w')
    # m = sympy.Matrix(w*F1+(1-w)*F2)
    #
    # coeff = (m.det().as_poly().coeffs())
    def fun(a): return np.linalg.det(a*F1+(1-a)*F2)
    a0 = fun(0)
    a1 = (fun(1) - fun(-1))/3-(fun(2)-fun(-2))/12
    a2 = 0.5*fun(1) + 0.5*fun(-1) - fun(0)
    a3 = (fun(1) - fun(-1))/6 + (fun(2) - fun(-2))/12

    coeff = [a3, a2, a1, a0]
    # print(coeff)
    # coeff = coeff[::-1]
    # print(coeff)
    soln = np.roots(coeff)
    soln = soln[np.isreal(soln)]
    # print(soln)
    # print(soln.shape)
    Fs = []
    T = np.array([[1. / M, 0, 0], [0, 1. / M, 0], [0, 0, 1]])
    for i in range(len(soln)):
        F = (np.matmul(T.T, np.matmul((soln[i]*F1+(1-soln[i])*F2), T)))
        # F = helper.refineF(F,pts1,pts2)
        # F = helper._singularize(F)
        Fs.append(F)

    return Fs


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''


def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation

    return np.linalg.multi_dot([K2.T, F, K1])


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''


def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation

    N = pts1.shape[0]
    p1 = C1[0]
    p2 = C1[1]
    p3 = C1[2]

    _p1 = C2[0]
    _p2 = C2[1]
    _p3 = C2[2]

    P = []
    for i in range(N):
        x, y = list(pts1[i])
        _x, _y = list(pts2[i])

        A = np.array([y*p3-p2, p1-x*p3, _y*_p3-_p2, _p1-_x*_p3])
        _, _, VT = np.linalg.svd(A)
        P_i = VT[-1, :3]/VT[-1, -1]
        P.append(P_i)
    P = np.array(P)

    err = 0
    # Calculate reprojection err

    pts1_reproj = np.matmul(C1, np.concatenate(
        [P, np.ones((P.shape[0], 1))], axis=-1).T)
    pts1_reproj = (pts1_reproj[:2]/pts1_reproj[-1]).T  # Nx2

    pts2_reproj = np.matmul(C2, np.concatenate(
        [P, np.ones((P.shape[0], 1))], axis=-1).T)
    pts2_reproj = (pts2_reproj[:2]/pts2_reproj[-1]).T  # Nx2

    err = np.sum(np.linalg.norm(pts1-pts1_reproj, axis=-1) +
                 np.linalg.norm(pts2-pts2_reproj, axis=-1))

    return P, err


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''


def epipolarCorrespondence(im1, im2, F, x1, y1, w_size=5, differ=30):
    # Replace pass by your implementation
    x = np.array([x1, y1, 1]).reshape((3, 1))
    _epipolarLine = np.matmul(F, x).flatten()
    a, b, c = list(_epipolarLine)

    # find possible points
    im2_height, im2_width = im2.shape[:2]
    m = -a/b
    possible_points = []
    if abs(m) > 1:  # iterate through y
        for y in range(y1-differ, y1+differ):
            x = int(-b*y/a - c/a)

            if x < im2_width and x >= 0:
                possible_points.append([x, y])
    else:  # iterate through x
        for x in range(x1-differ, x1+differ):
            y = int(-a*x/b - c)
            if y < im2_height and y >= 0:
                possible_points.append([x, y])

    min_err = 1e6
    best_match = [0, 0]

    w_im1 = im1[y1-w_size//2:y1+w_size//2+1, x1-w_size//2:x1+w_size//2+1]
    for (x2, y2) in possible_points:
        if (x2 < w_size//2) or (x2 > im2_width-(w_size//2+1)) or (y2 < w_size//2) or (y2 > im2_height-(w_size//2+1)):
            continue
        w_im2 = im2[y2-w_size//2:y2+w_size//2+1, x2-w_size//2:x2+w_size//2+1]

        g = gaussian_filter_2d(w_size)
        diff = np.sqrt(np.sum(np.square(w_im1-w_im2), axis=-1))
        err = np.mean(np.multiply(g, diff))

        if err < min_err:
            min_err = err
            best_match = [x2, y2]
    return best_match


'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''


def ransacF(pts1, pts2, M, n_iter=200, threshold=1e-3):
    # Replace pass by your implementation
    num_samples = pts1.shape[0]
    h_pts1 = np.concatenate([pts1, np.ones((pts1.shape[0], 1))], axis=1)
    h_pts2 = np.concatenate([pts2, np.ones((pts2.shape[0], 1))], axis=1)

    max_inliers = np.zeros(num_samples)
    max_num_inliers = 0

    for _ in tqdm(range(n_iter)):
        selected_idx = np.random.choice(num_samples, 7, replace=False)
        selected_pts1 = pts1[selected_idx]
        selected_pts2 = pts2[selected_idx]
        Fs = sevenpoint(selected_pts1, selected_pts2, M)

        for F in Fs:
            eq = np.linalg.multi_dot([h_pts2, F, h_pts1.T])
            eq = np.abs(np.diag(eq))
            inliers = np.where(eq < threshold, 1, 0)
            n_inliers = np.sum(inliers)

            if n_inliers > max_num_inliers:
                max_num_inliers = n_inliers
                max_inliers = inliers
    inlier_indices = np.argwhere(max_inliers == 1).flatten()
    F = eightpoint(pts1[inlier_indices], pts2[inlier_indices], M)
    return F, max_inliers.reshape((num_samples, 1))


'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
    Hint: Find reference in ../rodrigues.pdf
'''


def rodrigues(r):
    # Replace pass by your implementation
    theta = np.linalg.norm(r)
    u = r/theta
    c = np.cos(theta)
    s = np.sin(theta)
    I = np.identity(3)

    u_x, u_y, u_z = list(u.flatten())
    skew_u = np.array([[0, -u_z, u_y], [u_z, 0, -u_x], [-u_y, u_x, 0]])
    R = c*I + (1-c)*np.matmul(u, u.T) + s*skew_u
    return R


'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''


def invRodrigues(R):
    # Replace pass by your implementation
    A = (R - R.T)/2
    rho = np.array([A[2, 1], A[0, 2], A[1, 0]]).reshape((3, 1))
    s = np.linalg.norm(rho)
    c = (np.sum(np.diag(R))-1)/2

    if s == 0 and c == 1:
        return np.zeros((3,1))
    if s == 0 and c == -1:
        tmp = R + np.identity(3)
        # v is a non-zero column of R+I
        v = np.zeros(3)
        if tmp[:, 0] != np.zeros(3):
            v = tmp[:, 0]
        elif tmp[:, 1] != np.zeros(3):
            v = tmp[:, 1]
        else:
            v = tmp[:, 2]
        u = v/np.linalg.norm(v)

        r = np.pi*u

        if ((r[0] == 0 and r[1] == 0 and r[2] < 0) or (r[0] == 0 and r[1] < 0) or (r[0] < 0)):
            r = -1*r
            return r.reshape((3,1))

        return r.reshape((3,1))
    if s != 0:
        u = rho/s
        theta = np.arctan2(s,c)
        r = u*theta
        return r.reshape((3,1))



'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''


def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    N = p1.shape[0]
    P = x[:-6].reshape((N, 3))
    P = np.concatenate([P, np.ones((N,1))], axis=-1)

    r2 = x[-6:-3].reshape((3,1))
    R2 = rodrigues(r2)
    t2 = x[-3:].reshape((3,1))
    M2 = np.concatenate([R2, t2], axis=-1)

    C1 = np.matmul(K1, M1)
    C2 = np.matmul(K2, M2)

    p1_hat = np.matmul(C1, P.T)
    p1_hat = (p1_hat[:2]/p1_hat[2]).T

    p2_hat = np.matmul(C2, P.T)
    p2_hat = (p2_hat[:2]/p2_hat[2]).T

    residuals = np.concatenate([(p1-p1_hat).reshape((-1)), (p2-p2_hat).reshape((-1))]).reshape((4*N, 1))
    
    return residuals

'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 2
            P2, the optimized 3D coordinates of points
'''


def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    _P_init = P_init.flatten()
    _r_init = invRodrigues(M2_init[:,:3]).flatten()
    _t_init = M2_init[:, 3].flatten()

    x_init = np.concatenate([_P_init, _r_init, _t_init])
    
    optimized_x, _ = leastsq(lambda x: (rodriguesResidual(K1, M1, p1, K2, p2, x).flatten()), x_init)
    
    N = p1.shape[0]
    P2 = optimized_x[:-6].reshape((N, 3))

    r2 = optimized_x[-6:-3].reshape((3,1))
    R2 = rodrigues(r2)
    t2 = optimized_x[-3:].reshape((3,1))
    M2 = np.concatenate([R2, t2], axis=-1)

    return M2, P2