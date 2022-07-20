# mercury
Unsupervised Point Cloud Pose Canonicalization By Approximating the Plane/s of Symmetry

![animation](https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-airplane-3-animation.gif?raw=true)

(Head over to the bottom portion of the page to view other results)

# Canonicalization
Given an object represented as a point set, or more commonly 
called **point cloud** in computer vision, $K \in [-1,1]^{s \times d}$ with $s$ 
number of $d$-dimensional points (specifically, we use $d=3$ here for 3-d points), two 
rotations $R_1, R_2 \in \mathbb{R}^{3\times 3}$, such that we obtain two 
rotations of the object $KR_1^T$ and $KR_2^T$, a 
**canonicalization**  $C(\cdot)$ is such that 

$$C(KR_1^T) = C(KR_2^T) = KR_3^T$$

for some rotation $R_3$. In other words, a canonicalization brings the object to a **canonical pose**.

# Motivation
This study was largely motivated by the ideas presented and articulated by Hinton on humans thinking in 
coordinate frames or instrinsic axes. In one of his talks, he said that humans have a 
preferred axis by which we frame objects, usually by a _long axis_ (a term we leave to define 
intuitively as an axis where the object is longest). Using this notion, we can identify the long axis 
of the object and canonicalize by rotating such that this long axis lies on any preferred axis 
(say the $y$ axis, or any other axis). 

Another take I took on this relied on the observation that most objects in real life have planes of symmetry 
that runs along the long axis of the object - and that maybe, we can canonicalize with a plane of 
symmetry instead of a long axis. Hence, this method relies on the assumption that objects in 
question would have planes of symmetry (or approximate symmetry) and that this plane is aligned 
for objects of the same class (e.g. airplane, sofa, table, etc.). And since this method relies on 
symmetry rather than semantically consistent features, this will not have the reliability of 
[canonical capsules](https://canonical-capsules.github.io/) which learns semantically consistent
segmentations, but this method does provide a direct method of canonicalization that 
satisfies the equation about and that is invariant to rotation without the need for training.

# Summary of Canonicalization
For a generic object $K$ finding the plane of symmetry is hard, 
but the planes of symmetry for an ellipsoid is well known. Knowing this, 
we can approximate an object $K$ by an ellipsoid $E$ and rotate the ellipsoid such 
that its minor(shortest), intermediate, and major axes lie on the $z$,$y$, and $x$ axes, 
respectively - and do the same rotation to $K$ resulting to $K^\prime$. 
Now we can study the symmetry $K^\prime$ along the planes that are normal to 
the canonical 3d basis unit vectors $\hat{i}, \hat{j}, \hat{k} \in {0, 1}^{3 \times 3}$. 
The plane (identified by the unit vector that is normal to that plane) that maximizes the 
symmetry is then oriented on $zx$ plane resulting to our 
canonicalized pose for the object $C(K)$.

> Note: I may add more (technical/mathematical) details soon, but the gist of it is using the ellipsoid resulting from 
> a PCA (which also is related to how this canonicalization is invariant to rotation of the point cloud)

## Canonicalization to a Reference
The above canonicalization works on objects with symmetries, but it does not guarantee 
semantic canonicalization - a concrete example is that a plane canonicalized by the above 
ends up with its cockpit and tail on the $y$ axis, but for some airplanes their tails can 
end up on the $-y$ direction and some on the $+y$ direction. To mitigate this, I also present 
a canonicalization to a reference - nevertheless, this method does not know about the semantics 
of object parts and do is not perfect. 

To canonicalize to an object $K$ to a reference $K_\textnormal{ref}$(already in canonical pose), we 
canonicalize $K$ as $C(K)$ and find a rotation of $C(K)$ that minimizes its chamfer distance with
$K_\textnormal{ref}$ - these rotations are constrained such that the rotation can only be 
$R_\alpha R_\beta R_\gamma$ where $\alpha \in \{0, \pi\}$ is the yaw angle, 
$\beta \in \{0, 0.5\pi, \pi\}$ is the pitch angle, and $\gamma \in \{0, \pi\}$ is the roll angle.

# Results
Here are rotating objects and their canonicalized counter part, they are canonicalized to a reference 
where, for each, set of object of the same class, the first of such object is canonicalized to a 
pose to serve as the reference of the rest of the objects within the same class.  

This dataset is of the shapenet dataset as a point cloud hosted [here](https://github.com/antao97/PointCloudDatasets).

## Airplane
<p>
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-airplane-0-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-airplane-1-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-airplane-2-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-airplane-3-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-airplane-4-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-airplane-5-animation.gif?raw=true" width="250">
</p>

## Bench
<p>
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-bench-0-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-bench-1-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-bench-2-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-bench-3-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-bench-4-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-bench-5-animation.gif?raw=true" width="250">
</p>

## Cabinet
<p>
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-cabinet-0-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-cabinet-1-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-cabinet-2-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-cabinet-3-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-cabinet-4-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-cabinet-5-animation.gif?raw=true" width="250">
</p>

## Chair
<p>
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-chair-0-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-chair-1-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-chair-2-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-chair-3-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-chair-4-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-chair-5-animation.gif?raw=true" width="250">
</p>

## Guitar
<p>
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-guitar-0-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-guitar-1-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-guitar-2-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-guitar-3-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-guitar-4-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-guitar-5-animation.gif?raw=true" width="250">
</p>

## Mug 
<p>
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-mug-0-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-mug-1-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-mug-2-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-mug-3-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-mug-4-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-mug-5-animation.gif?raw=true" width="250">
</p>

## Rifle
<p>
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-rifle-0-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-rifle-1-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-rifle-2-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-rifle-3-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-rifle-4-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-rifle-5-animation.gif?raw=true" width="250">
</p>

## Sofa
<p>
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-sofa-0-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-sofa-1-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-sofa-2-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-sofa-3-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-sofa-4-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-sofa-5-animation.gif?raw=true" width="250">
</p>

## Speaker
<p>
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-speaker-0-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-speaker-1-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-speaker-2-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-speaker-3-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-speaker-4-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-speaker-5-animation.gif?raw=true" width="250">
</p>

## Table
<p>
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-table-0-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-table-1-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-table-2-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-table-3-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-table-4-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-table-5-animation.gif?raw=true" width="250">
</p>

## Telephone
<p>
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-telephone-0-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-telephone-1-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-telephone-2-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-telephone-3-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-telephone-4-animation.gif?raw=true" width="250">
    <img src="https://github.com/mzguntalan/mercury-demo-files/blob/main/demo-animations/demo-telephone-5-animation.gif?raw=true" width="250">
</p>

# Dependencies
This projects depends on jax and numpy for computation, h5py for reading the datasets, tqdm, matplotlib, and imageio for creating the animations.
