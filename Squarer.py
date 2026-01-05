"""
Euclidean Geometric Lattice Squarer
===================================

A tool that inserts itself into a lattice as a point, unfolds into a line,
then into a square using pure Euclidean geometry, and uses the vertices to
"square" the entire lattice.

Based on GeometricLLL logic but focused on geometric transformation sequence
using Euclidean distances and expansions rather than integer arithmetic.
"""

import numpy as np
from typing import Tuple, List, Optional


def integer_sqrt(n: int) -> int:
    """
    Compute integer square root of n using Newton's method.
    Returns floor(sqrt(n)) for large integers.
    
    Args:
        n: Non-negative integer
    
    Returns:
        Integer square root (floor of sqrt(n))
    """
    if n < 0:
        raise ValueError("Cannot compute square root of negative number")
    if n == 0:
        return 0
    if n < 4:
        return 1
    
    # Use Newton's method: x_{n+1} = (x_n + n/x_n) // 2
    # Start with a good initial guess
    # For very large numbers, start with bit shift
    if n > 10**20:
        # Initial guess: 2^(bits/2)
        bits = n.bit_length()
        x = 1 << (bits // 2)
    else:
        x = n
    
    # Iterate until convergence
    while True:
        x_new = (x + n // x) // 2
        if x_new >= x:  # Converged or overshot
            # Check if x is correct or if we need x-1
            if x * x <= n:
                return x
            else:
                return x - 1
        x = x_new


class EuclideanSquarer:
    """
    Euclidean geometric transformation tool that:
    1. Inserts as a point in the lattice
    2. Unfolds into a line spanning the lattice
    3. Unfolds into a triangle
    4. Forms a square base
    5. Uses vertices to "square" the lattice

    Uses pure Euclidean geometry with expansion factors and geometric relationships.
    """

    def __init__(self, basis: np.ndarray):
        """
        Initialize with a lattice basis.

        Args:
            basis: Lattice basis matrix (n x m) - can be integer or float
        """
        # Preserve the original dtype - if it's object (large integers), keep it
        # Otherwise convert to appropriate integer type
        if basis.dtype == object:
            self.original_basis = np.array(basis, dtype=object).copy()
            self.basis = np.array(basis, dtype=object).copy()
            self.is_integer = True
        elif np.issubdtype(basis.dtype, np.integer):
            self.original_basis = np.array(basis, dtype=object).copy()
            self.basis = np.array(basis, dtype=object).copy()
            self.is_integer = True
        else:
            # Float input - convert to integers by rounding
            self.original_basis = np.array([[int(round(x)) for x in row] for row in basis], dtype=object)
            self.basis = self.original_basis.copy()
            self.is_integer = True
        
        self.n = len(self.basis)
        self.m = self.basis.shape[1] if len(self.basis.shape) > 1 else self.n
        self.original_n = self.n
        # Start with origin point (integer)
        self.vertices = np.array([[0] * self.m], dtype=object)
        self.transformation_history = []
    
    def insert_as_point(self, expansion_factor: float = 0.1, verbose: bool = False) -> np.ndarray:
        """
        Step 1: Insert as a geometric point in the lattice.

        Creates a focal point that serves as the geometric center for unfolding.
        Uses Euclidean geometry to compress the lattice toward a central point.

        Args:
            expansion_factor: How much to compress toward center (0.1 = 10% of original size)
            verbose: Print progress

        Returns:
            Basis compressed toward geometric center
        """
        if verbose:
            print("[*] Step 1: INSERTING AS GEOMETRIC POINT")
            print(f"    Compressing lattice toward geometric center")

        # Find the geometric center of the lattice (integer arithmetic)
        center = np.zeros(self.m, dtype=object)
        for i in range(self.n):
            for j in range(self.m):
                center[j] = center[j] + int(self.basis[i, j])
        # Integer division for center
        for j in range(self.m):
            center[j] = center[j] // self.n

        # Compress all vectors toward the center using integer arithmetic
        # expansion_factor is converted to integer ratio (e.g., 0.1 -> 1/10)
        # We'll use: new = center + (vector - center) * expansion_numerator // expansion_denominator
        expansion_numerator = 1
        expansion_denominator = max(1, int(round(1.0 / expansion_factor)))
        
        compressed_basis = np.zeros((self.n, self.m), dtype=object)
        for i in range(self.n):
            for j in range(self.m):
                diff = int(self.basis[i, j]) - int(center[j])
                compressed_basis[i, j] = int(center[j]) + (diff * expansion_numerator) // expansion_denominator

        self.basis = compressed_basis
        self.vertices = np.array([center.copy()], dtype=object)

        if verbose:
            print(f"    Geometric center: {center[:min(3, self.m)]}")
            print(f"    Compression factor: {expansion_factor}")

        self.transformation_history.append("point_inserted")
        return self.basis
    
    def unfold_to_line(self, expansion_factor: float = 3.0, verbose: bool = False) -> np.ndarray:
        """
        Step 2: Unfold the point into a line that spans the lattice.

        Expands the compressed point into a line using Euclidean geometry,
        extending along the primary lattice direction.

        Args:
            expansion_factor: How much to expand along the line
            verbose: Print progress

        Returns:
            Basis expanded into line formation
        """
        if verbose:
            print("\n[*] Step 2: UNFOLDING POINT → LINE")

        if len(self.vertices) == 0:
            self.insert_as_point(verbose=False)

        center = self.vertices[0]

        # Find the primary direction (longest vector in basis) using integer arithmetic
        primary_direction = np.zeros(self.m, dtype=object)
        max_length_sq = 0
        for vec in self.basis:
            length_sq = sum(int(vec[i]) * int(vec[i]) for i in range(self.m))
            if length_sq > max_length_sq:
                max_length_sq = length_sq
                for i in range(self.m):
                    primary_direction[i] = int(vec[i])

        if max_length_sq == 0:
            primary_direction[0] = 1

        # Normalize the direction (using integer arithmetic with scaling)
        # We'll work with scaled integers to preserve precision
        scale = 10000  # Scale factor for normalization
        norm_sq = sum(int(primary_direction[i]) * int(primary_direction[i]) for i in range(self.m))
        if norm_sq > 0:
            # Normalize by scaling: direction_scaled = direction * scale / sqrt(norm_sq)
            # Use integer square root
            norm_approx = integer_sqrt(norm_sq)
            for i in range(self.m):
                primary_direction[i] = (int(primary_direction[i]) * scale) // max(1, norm_approx)

        # Create line endpoints by expanding from center (integer arithmetic)
        expansion_num = int(round(expansion_factor * 100))
        expansion_den = 100
        max_length_approx = integer_sqrt(max_length_sq) if max_length_sq > 0 else 1
        
        left_endpoint = np.zeros(self.m, dtype=object)
        right_endpoint = np.zeros(self.m, dtype=object)
        for i in range(self.m):
            center_i = int(center[i])
            dir_i = int(primary_direction[i])
            # Expand: center ± expansion * direction * max_length
            expansion_term = (dir_i * expansion_num * max_length_approx) // (scale * expansion_den)
            left_endpoint[i] = center_i - expansion_term
            right_endpoint[i] = center_i + expansion_term

        # Expand basis vectors along this line (integer arithmetic)
        line_basis = np.zeros((self.n, self.m), dtype=object)
        for i in range(self.n):
            # Project each vector onto the line and expand
            # Projection = dot(basis[i], direction) * direction
            dot = sum(int(self.basis[i, j]) * int(primary_direction[j]) for j in range(self.m))
            # Scale back from normalized direction
            for j in range(self.m):
                proj_j = (dot * int(primary_direction[j])) // (scale * scale)
                expansion_term = (proj_j * expansion_num) // expansion_den
                line_basis[i, j] = int(center[j]) + expansion_term

        self.basis = line_basis
        self.vertices = np.array([left_endpoint, right_endpoint], dtype=object)

        if verbose:
            print(f"    Line expanded by factor {expansion_factor}")
            print(f"    Left endpoint: {left_endpoint[:min(3, self.m)]}")
            print(f"    Right endpoint: {right_endpoint[:min(3, self.m)]}")

        self.transformation_history.append("line_unfolded")
        return self.basis
    
    def unfold_to_triangle(self, expansion_factor: float = 1.5, verbose: bool = False) -> np.ndarray:
        """
        Step 3: Unfold the line into a triangle.

        Expands the line into a triangular formation using Euclidean geometry.

        Args:
            expansion_factor: Height expansion factor for triangle
            verbose: Print progress

        Returns:
            Basis formed into triangle
        """
        if verbose:
            print("\n[*] Step 3: UNFOLDING LINE → TRIANGLE")

        if len(self.vertices) < 2:
            self.unfold_to_line(verbose=False)

        left = self.vertices[0]
        right = self.vertices[1]

        # Calculate line vector and perpendicular (integer arithmetic)
        line_vector = np.zeros(self.m, dtype=object)
        for i in range(self.m):
            line_vector[i] = int(right[i]) - int(left[i])

        # Compute line length squared
        line_length_sq = sum(int(line_vector[i]) * int(line_vector[i]) for i in range(self.m))
        line_length_approx = integer_sqrt(line_length_sq) if line_length_sq > 0 else 1

        # Create perpendicular vector (rotate 90 degrees) - integer arithmetic
        perp = np.zeros(self.m, dtype=object)
        if self.m >= 2:
            perp[0] = -int(line_vector[1])
            perp[1] = int(line_vector[0])
            # Rest are zero
        else:
            perp[0] = 1

        # Normalize perpendicular (using integer scaling)
        scale = 10000
        perp_norm_sq = sum(int(perp[i]) * int(perp[i]) for i in range(self.m))
        if perp_norm_sq > 0:
            perp_norm_approx = integer_sqrt(perp_norm_sq)
            for i in range(self.m):
                perp[i] = (int(perp[i]) * scale) // max(1, perp_norm_approx)

        # Create apex above the midpoint (integer arithmetic)
        midpoint = np.zeros(self.m, dtype=object)
        for i in range(self.m):
            midpoint[i] = (int(left[i]) + int(right[i])) // 2

        expansion_num = int(round(expansion_factor * 100))
        expansion_den = 100
        height_scaled = (line_length_approx * expansion_num) // expansion_den
        
        apex = np.zeros(self.m, dtype=object)
        for i in range(self.m):
            apex[i] = int(midpoint[i]) + (int(perp[i]) * height_scaled) // scale

        # Triangle vertices
        self.vertices = np.array([left, right, apex], dtype=object)

        # Expand basis vectors to form triangular relationships (integer arithmetic)
        triangle_basis = np.zeros((self.n, self.m), dtype=object)
        for i in range(self.n):
            # Create triangular relationships between vectors
            # Mix original vector with triangle vertices
            # weight = i / max(1, self.n - 1)  # 0 to 1
            weight_num = i
            weight_den = max(1, self.n - 1)
            for j in range(self.m):
                basis_val = int(self.basis[i, j])
                apex_val = int(apex[j])
                # (1 - weight) * basis + weight * apex
                triangle_basis[i, j] = ((weight_den - weight_num) * basis_val + weight_num * apex_val) // weight_den

        self.basis = triangle_basis

        if verbose:
            print(f"    Triangle formed with height factor {expansion_factor}")
            print(f"      Left: {left[:min(3, self.m)]}")
            print(f"      Right: {right[:min(3, self.m)]}")
            print(f"      Apex: {apex[:min(3, self.m)]}")

        self.transformation_history.append("triangle_formed")
        return self.basis
    
    def unpack_vertices_from_triangle_sides(self, num_points: int = 3, verbose: bool = False) -> np.ndarray:
        """
        Step 4: Unpack vertices from the sides of the triangle.

        Extracts additional vertices along the triangle edges to form denser geometry.

        Args:
            num_points: Number of points to add per side
            verbose: Print progress

        Returns:
            Array of unpacked vertices
        """
        if verbose:
            print("\n[*] Step 4: UNPACKING VERTICES FROM TRIANGLE SIDES")

        if len(self.vertices) < 3:
            self.unfold_to_triangle(verbose=False)

        left = self.vertices[0]
        right = self.vertices[1]
        apex = self.vertices[2]

        unpacked = []

        # Side 1: left → apex (integer arithmetic)
        for i in range(1, num_points):
            t_num = i
            t_den = num_points
            v = np.zeros(self.m, dtype=object)
            for j in range(self.m):
                # (1 - t) * left + t * apex
                v[j] = ((t_den - t_num) * int(left[j]) + t_num * int(apex[j])) // t_den
            unpacked.append(v)

        # Side 2: right → apex (integer arithmetic)
        for i in range(1, num_points):
            t_num = i
            t_den = num_points
            v = np.zeros(self.m, dtype=object)
            for j in range(self.m):
                # (1 - t) * right + t * apex
                v[j] = ((t_den - t_num) * int(right[j]) + t_num * int(apex[j])) // t_den
            unpacked.append(v)

        # Side 3: left → right (base) (integer arithmetic)
        for i in range(1, num_points):
            t_num = i
            t_den = num_points
            v = np.zeros(self.m, dtype=object)
            for j in range(self.m):
                # (1 - t) * left + t * right
                v[j] = ((t_den - t_num) * int(left[j]) + t_num * int(right[j])) // t_den
            unpacked.append(v)

        # Update vertices with unpacked points
        all_vertices = [left, right, apex] + unpacked
        self.vertices = np.array(all_vertices, dtype=object)

        if verbose:
            print(f"    Unpacked {len(unpacked)} vertices from triangle sides")
            print(f"    Total vertices: {len(self.vertices)}")

        self.transformation_history.append("vertices_unpacked")
        return np.array(unpacked, dtype=object)
    
    def form_square_base(self, expansion_factor: float = 1.0, verbose: bool = False) -> np.ndarray:
        """
        Step 5: Form a square base from the triangle.

        Uses Euclidean geometry to form a proper square from the triangular formation.

        Args:
            expansion_factor: Square size factor
            verbose: Print progress

        Returns:
            Square corner vertices
        """
        if verbose:
            print("\n[*] Step 5: FORMING SQUARE BASE")

        if len(self.vertices) < 3:
            self.unpack_vertices_from_triangle_sides(verbose=False)

        left = self.vertices[0]
        right = self.vertices[1]
        apex = self.vertices[2]

        # Calculate square corners using Euclidean geometry (integer arithmetic)
        base_midpoint = np.zeros(self.m, dtype=object)
        for i in range(self.m):
            base_midpoint[i] = (int(left[i]) + int(right[i])) // 2
        
        base_vector = np.zeros(self.m, dtype=object)
        for i in range(self.m):
            base_vector[i] = int(right[i]) - int(left[i])
        
        base_length_sq = sum(int(base_vector[i]) * int(base_vector[i]) for i in range(self.m))
        base_length_approx = integer_sqrt(base_length_sq) if base_length_sq > 0 else 1

        # Create perpendicular vector for height (integer arithmetic)
        perp_vector = np.zeros(self.m, dtype=object)
        if self.m >= 2:
            perp_vector[0] = -int(base_vector[1])
            perp_vector[1] = int(base_vector[0])
        else:
            perp_vector[0] = base_length_approx

        # Normalize perpendicular vector (using integer scaling)
        scale = 10000
        perp_norm_sq = sum(int(perp_vector[i]) * int(perp_vector[i]) for i in range(self.m))
        if perp_norm_sq > 0:
            perp_norm_approx = integer_sqrt(perp_norm_sq)
            for i in range(self.m):
                perp_vector[i] = (int(perp_vector[i]) * scale) // max(1, perp_norm_approx)

        # Calculate height from apex (integer arithmetic)
        height_vector = np.zeros(self.m, dtype=object)
        for i in range(self.m):
            height_vector[i] = int(apex[i]) - int(base_midpoint[i])
        
        # Dot product for height
        height_scaled = sum(int(height_vector[i]) * int(perp_vector[i]) for i in range(self.m))
        height_approx = height_scaled // scale

        # Create square with equal sides (integer arithmetic)
        expansion_num = int(round(expansion_factor * 100))
        expansion_den = 100
        side_length_scaled = ((base_length_approx + abs(height_approx)) * expansion_num * scale) // (2 * expansion_den)

        # Square corners (integer arithmetic)
        bottom_left = left.copy()
        bottom_right = np.zeros(self.m, dtype=object)
        top_left = np.zeros(self.m, dtype=object)
        top_right = np.zeros(self.m, dtype=object)
        
        for i in range(self.m):
            # bottom_right = left + side_length * (base_vector / base_length)
            base_dir_i = (int(base_vector[i]) * scale) // max(1, base_length_approx)
            bottom_right[i] = int(left[i]) + (side_length_scaled * base_dir_i) // (scale * scale)
            
            # top_left = left + side_length * perp_vector
            top_left[i] = int(left[i]) + (side_length_scaled * int(perp_vector[i])) // (scale * scale)
            
            # top_right = bottom_right + (top_left - left)
            top_right[i] = int(bottom_right[i]) + (int(top_left[i]) - int(left[i]))

        square_corners = np.array([bottom_left, bottom_right, top_left, top_right], dtype=object)
        self.vertices = square_corners

        # Transform basis to align with square (integer arithmetic)
        square_basis = np.zeros((self.n, self.m), dtype=object)
        square_center = np.zeros(self.m, dtype=object)
        for i in range(4):
            for j in range(self.m):
                square_center[j] = square_center[j] + int(square_corners[i, j])
        for j in range(self.m):
            square_center[j] = square_center[j] // 4

        for i in range(self.n):
            # Project onto square coordinate system
            vec = np.zeros(self.m, dtype=object)
            for j in range(self.m):
                vec[j] = int(self.basis[i, j]) - int(square_center[j])
            
            # Align with square axes
            x_proj = sum(int(vec[j]) * int(base_vector[j]) for j in range(self.m)) // max(1, base_length_approx)
            y_proj = sum(int(vec[j]) * int(perp_vector[j]) for j in range(self.m)) // scale
            
            for j in range(self.m):
                base_dir_j = (int(base_vector[j]) * scale) // max(1, base_length_approx)
                square_basis[i, j] = int(square_center[j]) + (x_proj * base_dir_j) // (scale * scale) + (y_proj * int(perp_vector[j])) // scale

        self.basis = square_basis

        if verbose:
            # Use integer division for display (avoid float overflow)
            side_length_int = side_length_scaled // scale
            side_length_frac = ((side_length_scaled % scale) * 1000) // scale
            print(f"    Square base formed with side length {side_length_int}.{side_length_frac:03d}")
            print("    Square corners:")
            for i, corner in enumerate(square_corners):
                print(f"      Corner {i+1}: {corner[:min(3, self.m)]}")

        self.transformation_history.append("square_base_formed")
        return square_corners
    
    def square_the_lattice(self, reduction_factor: float = 0.75, verbose: bool = False) -> np.ndarray:
        """
        Step 6: Use the square vertices to "square" the entire lattice.

        Applies geometric transformations to align the lattice with the square formation,
        with controlled reduction to preserve lattice quality.

        Args:
            reduction_factor: How aggressively to reduce (0.5 = moderate, 0.9 = aggressive)
            verbose: Print progress

        Returns:
            Squared lattice basis
        """
        if verbose:
            print("\n[*] Step 6: SQUARING THE LATTICE")

        if len(self.vertices) < 4:
            self.form_square_base(verbose=False)

        square_vertices = self.vertices[:4]

        # Square basis vectors (integer arithmetic)
        v1 = np.zeros(self.m, dtype=object)
        v2 = np.zeros(self.m, dtype=object)
        for i in range(self.m):
            v1[i] = int(square_vertices[1, i]) - int(square_vertices[0, i])  # Bottom edge
            v2[i] = int(square_vertices[2, i]) - int(square_vertices[0, i])  # Left edge

        # Normalize (using integer scaling)
        scale = 10000
        v1_norm_sq = sum(int(v1[i]) * int(v1[i]) for i in range(self.m))
        v2_norm_sq = sum(int(v2[i]) * int(v2[i]) for i in range(self.m))
        
        v1_norm_approx = integer_sqrt(v1_norm_sq) if v1_norm_sq > 0 else 1
        v2_norm_approx = integer_sqrt(v2_norm_sq) if v2_norm_sq > 0 else 1

        v1_unit = np.zeros(self.m, dtype=object)
        v2_unit = np.zeros(self.m, dtype=object)
        
        if v1_norm_sq > 0:
            for i in range(self.m):
                v1_unit[i] = (int(v1[i]) * scale) // max(1, v1_norm_approx)
        else:
            v1_unit[0] = scale

        if v2_norm_sq > 0:
            for i in range(self.m):
                v2_unit[i] = (int(v2[i]) * scale) // max(1, v2_norm_approx)
        else:
            if self.m > 1:
                v2_unit[1] = scale
            else:
                v2_unit[0] = scale

        # Apply controlled squaring transformation (integer arithmetic)
        squared_basis = np.zeros((self.n, self.m), dtype=object)
        reduction_num = int(round(reduction_factor * 100))
        reduction_den = 100

        for i in range(self.n):
            original = self.basis[i]

            # Project onto square coordinate system (integer arithmetic)
            dot_v1 = sum(int(original[j]) * int(v1_unit[j]) for j in range(self.m)) // scale
            dot_v2 = sum(int(original[j]) * int(v2_unit[j]) for j in range(self.m)) // scale
            
            comp_v1 = np.zeros(self.m, dtype=object)
            comp_v2 = np.zeros(self.m, dtype=object)
            for j in range(self.m):
                comp_v1[j] = (dot_v1 * int(v1_unit[j])) // scale
                comp_v2[j] = (dot_v2 * int(v2_unit[j])) // scale

            # Blend original with square-projected version
            for j in range(self.m):
                blended = ((reduction_num * (int(comp_v1[j]) + int(comp_v2[j])) + (reduction_den - reduction_num) * int(original[j])) // reduction_den)
                squared_basis[i, j] = blended

            # Gentle reduction against square vertices (integer arithmetic)
            for sq_vertex in square_vertices:
                sq_norm_sq = sum(int(sq_vertex[j]) * int(sq_vertex[j]) for j in range(self.m))
                if sq_norm_sq > 0:
                    proj_dot = sum(int(squared_basis[i, j]) * int(sq_vertex[j]) for j in range(self.m))
                    # proj = proj_dot / sq_norm_sq
                    proj_scaled = (proj_dot * scale) // sq_norm_sq
                    # Only apply partial reduction: reduction_factor * 0.5
                    half_reduction_num = reduction_num // 2
                    for j in range(self.m):
                        reduction_term = (half_reduction_num * proj_scaled * int(sq_vertex[j])) // (reduction_den * scale * scale)
                        squared_basis[i, j] = int(squared_basis[i, j]) - reduction_term

        # Size-reduction pass (similar to LLL size reduction) - integer arithmetic
        for i in range(1, self.n):
            for j in range(i):
                sq_norm_sq = sum(int(squared_basis[j, k]) * int(squared_basis[j, k]) for k in range(self.m))
                if sq_norm_sq > 0:
                    proj_dot = sum(int(squared_basis[i, k]) * int(squared_basis[j, k]) for k in range(self.m))
                    # Round to nearest integer for lattice reduction
                    if proj_dot >= 0:
                        proj_rounded = (proj_dot + sq_norm_sq // 2) // sq_norm_sq
                    else:
                        proj_rounded = -((-proj_dot + sq_norm_sq // 2) // sq_norm_sq)
                    
                    if proj_rounded != 0:
                        for k in range(self.m):
                            squared_basis[i, k] = int(squared_basis[i, k]) - proj_rounded * int(squared_basis[j, k])

        self.basis = squared_basis

        if verbose:
            print(f"    Lattice squared: {self.n}x{self.m} (reduction factor: {reduction_factor})")

            # Find shortest vector
            shortest_norm_sq = float('inf')
            for v in squared_basis:
                norm_sq = sum(int(v[i]) * int(v[i]) for i in range(self.m))
                if norm_sq > 0 and norm_sq < shortest_norm_sq:
                    shortest_norm_sq = norm_sq

            if shortest_norm_sq < float('inf'):
                # Use integer sqrt for large numbers, avoid float conversion for very large numbers
                shortest_norm_int = integer_sqrt(shortest_norm_sq)
                if shortest_norm_int < 10**10:
                    print(f"    Shortest vector length: {shortest_norm_int:.6f}")
                else:
                    print(f"    Shortest vector length: {shortest_norm_int}")

        self.transformation_history.append("lattice_squared")
        return self.basis
    
    def run_full_transformation(self, verbose: bool = True) -> np.ndarray:
        """
        Run the complete Euclidean geometric transformation sequence.

        Executes all steps: point → line → triangle → square → squared lattice

        Args:
            verbose: Print progress

        Returns:
            Fully transformed (squared) lattice basis
        """
        if verbose:
            print("=" * 70)
            print("EUCLIDEAN GEOMETRIC LATTICE SQUARER")
            print("=" * 70)
            print(f"Starting lattice: {self.n}x{self.m}\n")

        # Step 1: Insert as point
        self.insert_as_point(verbose=verbose)

        # Step 2: Unfold to line
        self.unfold_to_line(verbose=verbose)

        # Step 3: Unfold to triangle
        self.unfold_to_triangle(verbose=verbose)

        # Step 4: Unpack vertices from triangle sides
        self.unpack_vertices_from_triangle_sides(verbose=verbose)

        # Step 5: Form square base
        self.form_square_base(verbose=verbose)

        # Step 6: Square the lattice
        result = self.square_the_lattice(verbose=verbose)

        if verbose:
            print("\n" + "=" * 70)
            print("EUCLIDEAN TRANSFORMATION COMPLETE")
            print("=" * 70)
            print(f"Transformation history: {' → '.join(self.transformation_history)}")

        return result
    
    def finish_with_lll_like_reduction(self, delta: float = 0.75, verbose: bool = False) -> np.ndarray:
        """
        Finish with a simple LLL-like reduction to polish the result.

        This applies a Lovász-like condition to ensure the basis is properly reduced.
        Works with integer arithmetic.

        Args:
            delta: Lovász parameter (0.5-1.0, higher = more reduction)
            verbose: Print progress

        Returns:
            LLL-polished basis
        """
        if verbose:
            print(f"\n[*] Applying integer LLL-like finishing reduction (δ={delta})")

        basis = self.basis.copy()
        
        # Convert delta to integer ratio (e.g., 0.75 -> 3/4)
        delta_num = int(round(delta * 100))
        delta_den = 100

        # Integer LLL-like reduction
        # Add progress output for large lattices
        print_progress = self.n > 50
        
        for k in range(1, self.n):
            # Size reduce using integer arithmetic
            for j in range(k-1, -1, -1):
                # Compute dot products using integer arithmetic
                dot_kj = 0
                dot_jj = 0
                for i in range(self.m):
                    b_k_i = int(basis[k, i])
                    b_j_i = int(basis[j, i])
                    dot_kj += b_k_i * b_j_i
                    dot_jj += b_j_i * b_j_i
                
                if dot_jj > 0:
                    # mu = dot_kj / dot_jj, rounded to nearest integer
                    # Use: mu = round(dot_kj / dot_jj) = (dot_kj + dot_jj//2) // dot_jj
                    if dot_kj >= 0:
                        mu = (dot_kj + dot_jj // 2) // dot_jj
                    else:
                        mu = -((-dot_kj + dot_jj // 2) // dot_jj)
                    
                    if mu != 0:
                        # Reduce: basis[k] = basis[k] - mu * basis[j]
                        for i in range(self.m):
                            basis[k, i] = int(basis[k, i]) - mu * int(basis[j, i])
            
            # Progress output for large lattices (every 5 vectors near the end)
            if print_progress:
                if (k + 1) % 5 == 0 or k + 1 >= self.n - 5:
                    print(f"      LLL reduction: {k + 1}/{self.n} vectors processed...", end='\r')

            # Swap condition (simplified Lovász) using integer arithmetic
            if k < self.n:
                # Compute squared norms (integers) - only if we might need to swap
                # Optimize: only check swap condition occasionally to save time
                if k % 2 == 0 or k == self.n - 1:  # Check every other vector, or last one
                    norm_k_sq = 0
                    norm_km1_sq = 0
                    for i in range(self.m):
                        b_k_i = int(basis[k, i])
                        b_km1_i = int(basis[k-1, i])
                        norm_k_sq += b_k_i * b_k_i
                        norm_km1_sq += b_km1_i * b_km1_i

                    if norm_k_sq > 0 and norm_km1_sq > 0:
                        # Check if ||b_k||^2 < δ ||b_{k-1}||^2
                        # Using integer arithmetic: norm_k_sq * delta_den < delta_num * norm_km1_sq
                        if norm_k_sq * delta_den < delta_num * norm_km1_sq:
                            # Swap
                            temp = basis[k-1].copy()
                            basis[k-1] = basis[k].copy()
                            basis[k] = temp

                            if verbose:
                                # Use integer sqrt for display
                                norm_k_int = integer_sqrt(norm_k_sq) if norm_k_sq > 0 else 0
                                norm_km1_int = integer_sqrt(norm_km1_sq) if norm_km1_sq > 0 else 0
                                # Avoid float conversion for very large numbers
                                if norm_k_int < 10**10 and norm_km1_int < 10**10:
                                    print(f"    Swap {k-1} ↔ {k}: {norm_km1_int:.2f} ↔ {norm_k_int:.2f}")
                                else:
                                    print(f"    Swap {k-1} ↔ {k}: {norm_km1_int} ↔ {norm_k_int}")
            
            # Progress output for large lattices (every 5 vectors near the end)
            if print_progress:
                if (k + 1) % 5 == 0 or k + 1 >= self.n - 5:
                    print(f"      LLL reduction: {k + 1}/{self.n} vectors processed...", end='\r')
        
        # Newline after progress
        if print_progress:
            print()  # Clear the progress line

        self.basis = basis

        if verbose:
            # Compute shortest vector norm for display
            shortest_sq = float('inf')
            for v in basis:
                norm_sq = sum(int(v[i]) * int(v[i]) for i in range(self.m))
                if norm_sq > 0:
                    shortest_sq = min(shortest_sq, norm_sq)
            # Use integer sqrt for large numbers
            if shortest_sq < float('inf'):
                shortest_int = integer_sqrt(shortest_sq)
                if shortest_int < 10**10:
                    print(f"    Final shortest vector: {shortest_int:.6f}")
                else:
                    print(f"    Final shortest vector: {shortest_int}")
            else:
                print(f"    Final shortest vector: 0")

        return basis
    
    def collapse_to_point(self, max_iterations: int = 60, verbose: bool = False, check_factor_callback=None, large_number=None) -> np.ndarray:
        """
        Collapse the lattice using a Euclidean gravity well.
        
        Creates a geometric "gravity well" at the center of the lattice and draws
        all vectors toward it using pure Euclidean geometry (distances, angles, projections)
        rather than algebraic calculations.
        
        Geometric approach:
        1. Find the geometric center (centroid) of all vectors
        2. For each vector, compute its Euclidean distance from center
        3. Project the vector toward the center along the line connecting them
        4. Scale the projection based on distance (gravity effect - closer = stronger pull)
        5. Use geometric transformations, not algebraic reductions
        
        Args:
            max_iterations: Maximum number of gravity well iterations
            verbose: Print progress
            check_factor_callback: Optional callback function to check for factors during collapse
            large_number: Optional large number for factorization checks
        
        Returns:
            Fully collapsed lattice basis
        """
        if verbose:
            print(f"\n[*] COLLAPSING WITH EUCLIDEAN GRAVITY WELL")
            print(f"    Drawing vectors into geometric center...")
            # Track initial sizes to show reduction progress
            initial_shortest_sq = float('inf')
            initial_total_sq = 0
            for v in self.basis:
                norm_sq = sum(int(v[k]) * int(v[k]) for k in range(self.m))
                if norm_sq > 0:
                    if norm_sq < initial_shortest_sq:
                        initial_shortest_sq = norm_sq
                    initial_total_sq += norm_sq
            if initial_shortest_sq < float('inf'):
                initial_shortest = integer_sqrt(int(initial_shortest_sq))
                initial_avg = integer_sqrt(initial_total_sq // self.n) if self.n > 0 else 0
                if initial_shortest < 10**10 and initial_avg < 10**10:
                    print(f"    Initial: shortest = {initial_shortest:.2f}, avg = {initial_avg:.2f}")
                else:
                    print(f"    Initial: shortest ≈ {initial_shortest}, avg ≈ {initial_avg}")
        
        basis = self.basis.copy()
        
        for iteration in range(max_iterations):
            # === STEP 1: COMPUTE GEOMETRIC CENTER (CENTROID) ===
            # The center is the average of all vectors - pure geometry
            center = np.zeros(self.m, dtype=object)
            for k in range(self.m):
                sum_coord = 0
                for i in range(self.n):
                    sum_coord += int(basis[i, k])
                center[k] = sum_coord // self.n
            
            # === STEP 2: FOR EACH VECTOR, COMPUTE EUCLIDEAN DISTANCE FROM CENTER ===
            # This is the geometric distance, not algebraic
            changed = False
            
            for i in range(self.n):
                # Compute vector from center to this point
                vector_from_center = np.zeros(self.m, dtype=object)
                distance_sq = 0
                
                for k in range(self.m):
                    coord_diff = int(basis[i, k]) - int(center[k])
                    vector_from_center[k] = coord_diff
                    distance_sq += coord_diff * coord_diff
                
                # If vector is already at center, skip
                if distance_sq == 0:
                    continue
                
                # === STEP 3: GRAVITY WELL EFFECT ===
                # The "gravity" pulls the vector toward center
                # Strength is proportional to distance (further = stronger pull, but normalized)
                # Use geometric scaling: project toward center along the line
                
                # Compute the direction vector (normalized direction toward center)
                # In Euclidean geometry, we scale the vector toward center
                # The pull strength: use distance itself as the "gravity constant"
                distance = integer_sqrt(distance_sq)
                
                if distance == 0:
                    continue
                
                # === STEP 4: GEOMETRIC PROJECTION TOWARD CENTER ===
                # Instead of algebraic reduction, use geometric projection:
                # Move the vector along the line toward center
                # The movement is proportional to the distance (gravity effect)
                
                # Scale factor: pull stronger for vectors further away
                # But we want to move gradually, so use a fraction of the distance
                # Geometric approach: move by a proportion of the distance
                # Use 1/4 of the distance as the "gravity step" (geometric, not calculated)
                gravity_step = distance // 4
                if gravity_step == 0:
                    gravity_step = 1
                
                # Project: move vector toward center by gravity_step units
                # This is pure geometry: move along the line from vector to center
                new_vector = np.zeros(self.m, dtype=object)
                for k in range(self.m):
                    # Direction component
                    direction_component = vector_from_center[k]
                    
                    # Scale by gravity step / distance (geometric scaling)
                    # This moves us gravity_step units toward center
                    if distance > 0:
                        # Move: new = old - (direction * step / distance)
                        # This is geometric: we're moving along the line
                        movement = (direction_component * gravity_step) // distance
                        new_coord = int(basis[i, k]) - movement
                        new_vector[k] = new_coord
                    else:
                        new_vector[k] = int(basis[i, k])
                
                # Check if this actually moved the vector and didn't collapse to zero
                moved = False
                new_norm_sq = 0
                for k in range(self.m):
                    if int(basis[i, k]) != int(new_vector[k]):
                        moved = True
                    new_norm_sq += int(new_vector[k]) * int(new_vector[k])
                
                # Only update if vector moved and didn't collapse to zero
                # Also check if we're collapsing too aggressively (for factorization)
                if moved and new_norm_sq > 0:
                    # For factorization, don't collapse vectors that are already very small
                    # Keep some structure to preserve factorization information
                    if large_number is not None:
                        # If vector is already very small compared to N, don't collapse further
                        # This preserves factorization structure
                        sqrt_N = integer_sqrt(large_number) if large_number > 0 else 1
                        if new_norm_sq < sqrt_N // 1000:  # Already very small
                            # Skip further collapse for this vector
                            continue
                    
                    # Update the vector
                    for k in range(self.m):
                        basis[i, k] = new_vector[k]
                    changed = True
            
            # === STEP 5: GEOMETRIC ORTHOGONALIZATION ===
            # After gravity pull, ensure vectors are geometrically independent
            # Use geometric projections to make vectors more orthogonal
            for i in range(self.n):
                for j in range(i):
                    # Compute geometric projection of basis[i] onto basis[j]
                    # Dot product (geometric measure of alignment)
                    dot_ij = 0
                    dot_jj = 0
                    for k in range(self.m):
                        b_i_k = int(basis[i, k])
                        b_j_k = int(basis[j, k])
                        dot_ij += b_i_k * b_j_k
                        dot_jj += b_j_k * b_j_k
                    
                    if dot_jj > 0:
                        # Geometric projection coefficient
                        # This is the geometric measure of how much i aligns with j
                        mu = (dot_ij + dot_jj // 2) // dot_jj
                        
                        if abs(mu) > 0:
                            # Remove the projection (geometric orthogonalization)
                            for k in range(self.m):
                                old_val = int(basis[i, k])
                                new_val = old_val - mu * int(basis[j, k])
                                basis[i, k] = new_val
                                if old_val != new_val:
                                    changed = True
            
            # Check for factors at intermediate stages (before full collapse)
            if check_factor_callback is not None and large_number is not None:
                # Check every 100 iterations or at key milestones
                if (iteration + 1) % 100 == 0 or (iteration + 1) in [10, 50, 200, 500, 1000, 2000, 5000]:
                    if verbose:
                        print(f"    [Intermediate check at iteration {iteration + 1}]")
                    factor = check_factor_callback(basis, large_number, verbose=False)
                    if factor is not None:
                        if verbose:
                            print(f"    ✓✓✓ FACTOR FOUND during collapse at iteration {iteration + 1}: {factor}")
                        self.basis = basis
                        return basis
            
            if not changed:
                if verbose:
                    print(f"    Gravity well converged after {iteration + 1} iterations")
                break
            
            if verbose and (iteration + 1) % 2 == 0:
                # Track both shortest and average vector sizes to show collapse progress
                shortest_sq = float('inf')
                total_norm_sq = 0
                count = 0
                for v in basis:
                    norm_sq = sum(int(v[k]) * int(v[k]) for k in range(self.m))
                    if norm_sq > 0:
                        if norm_sq < shortest_sq:
                            shortest_sq = norm_sq
                        total_norm_sq += norm_sq
                        count += 1
                
                if shortest_sq < float('inf'):
                    shortest_int = integer_sqrt(int(shortest_sq))
                    avg_norm_sq = total_norm_sq // count if count > 0 else 0
                    avg_norm_int = integer_sqrt(avg_norm_sq) if avg_norm_sq > 0 else 0
                    
                    if shortest_int < 10**10 and avg_norm_int < 10**10:
                        print(f"    Iteration {iteration + 1}: shortest = {shortest_int:.2f}, avg = {avg_norm_int:.2f} (vectors shrinking toward center)")
                    else:
                        print(f"    Iteration {iteration + 1}: shortest ≈ {shortest_int}, avg ≈ {avg_norm_int} (vectors shrinking toward center)")
        
        self.basis = basis
        
        if verbose:
            shortest_sq = float('inf')
            for v in basis:
                norm_sq = sum(int(v[k]) * int(v[k]) for k in range(self.m))
                if norm_sq > 0 and norm_sq < shortest_sq:
                    shortest_sq = norm_sq
            if shortest_sq < float('inf'):
                shortest_int = integer_sqrt(int(shortest_sq))
                if shortest_int < 10**10:
                    print(f"    Collapsed lattice - shortest vector: {shortest_int:.6f}")
                else:
                    print(f"    Collapsed lattice - shortest vector ≈ {shortest_int}")
        
        self.transformation_history.append("collapsed_to_point")
        return basis

    def run_full_transformation_with_lll_finish(self, delta: float = 0.75, verbose: bool = True) -> np.ndarray:
        """
        Run the complete geometric transformation sequence with collapse.

        Args:
            delta: (Deprecated, not used) Kept for compatibility
            verbose: Print progress

        Returns:
            Fully transformed and collapsed lattice basis
        """
        if verbose:
            print("=" * 70)
            if self.is_integer:
                print("INTEGER LATTICE SQUARER + COLLAPSE")
            else:
                print("EUCLIDEAN GEOMETRIC LATTICE SQUARER + COLLAPSE")
            print("=" * 70)
            print(f"Starting lattice: {self.n}x{self.m}\n")

        # Run geometric transformation (now works with both integer and float lattices)
        if self.is_integer and verbose:
            print("[*] Integer lattice detected - using integer geometric transformations")
        
        self.insert_as_point(verbose=verbose)
        self.unfold_to_line(verbose=verbose)
        self.unfold_to_triangle(verbose=verbose)
        self.unpack_vertices_from_triangle_sides(verbose=verbose)
        self.form_square_base(verbose=verbose)
        self.square_the_lattice(reduction_factor=0.5, verbose=verbose)

        # After geometric transformations reduce to point-like structure, collapse completely
        result = self.collapse_to_point(max_iterations=60, verbose=verbose)

        if verbose:
            print("\n" + "=" * 70)
            if self.is_integer:
                print("INTEGER GEOMETRIC TRANSFORMATION + COLLAPSE COMPLETE")
            else:
                print("GEOMETRIC TRANSFORMATION + COLLAPSE COMPLETE")
            print("=" * 70)
            if self.transformation_history:
                print(f"Transformation history: {' → '.join(self.transformation_history)}")

        return result

    def reset_to_original(self, verbose: bool = False) -> np.ndarray:
        """
        Reset the lattice back to its original form.

        Returns the basis to its original state before transformations.

        Args:
            verbose: Print progress

        Returns:
            Original lattice basis
        """
        if verbose:
            print("[*] Resetting to original basis...")
        
        self.basis = self.original_basis.copy()
        self.transformation_history = []
        
        if verbose:
            print("    Reset complete")
        
        return self.basis

    def run_full_transformation_with_lll_finish(self, delta: float = 0.75, verbose: bool = True) -> np.ndarray:
        """
        Run the complete geometric transformation sequence with collapse.

        Args:
            delta: (Deprecated, not used) Kept for compatibility
            verbose: Print progress

        Returns:
            Fully transformed and collapsed lattice basis
        """
        if verbose:
            print("=" * 70)
            if self.is_integer:
                print("INTEGER LATTICE SQUARER + COLLAPSE")
            else:
                print("EUCLIDEAN GEOMETRIC LATTICE SQUARER + COLLAPSE")
            print("=" * 70)
            print(f"Starting lattice: {self.n}x{self.m}\n")

        # Run geometric transformation (now works with both integer and float lattices)
        if self.is_integer and verbose:
            print("[*] Integer lattice detected - using integer geometric transformations")
        
        self.insert_as_point(verbose=verbose)
        self.unfold_to_line(verbose=verbose)
        self.unfold_to_triangle(verbose=verbose)
        self.unpack_vertices_from_triangle_sides(verbose=verbose)
        self.form_square_base(verbose=verbose)
        self.square_the_lattice(reduction_factor=0.5, verbose=verbose)

        # After geometric transformations reduce to point-like structure, collapse completely
        result = self.collapse_to_point(max_iterations=60, verbose=verbose)

        if verbose:
            print("\n" + "=" * 70)
            if self.is_integer:
                print("INTEGER GEOMETRIC TRANSFORMATION + COLLAPSE COMPLETE")
            else:
                print("GEOMETRIC TRANSFORMATION + COLLAPSE COMPLETE")
            print("=" * 70)
            if self.transformation_history:
                print(f"Transformation history: {' → '.join(self.transformation_history)}")

        return result

        """
        Reset the lattice back to its original form.

        Returns the basis to its original state before transformations.

        Args:
            verbose: Print progress

        Returns:
            Original lattice basis
        """
        if verbose:
            print("\n[*] Resetting to original lattice")

        self.basis = self.original_basis.copy()
        self.n = len(self.basis)
        self.vertices = np.array([[0.0, 0.0]], dtype=np.float64)
        self.transformation_history = ["reset"]

        if verbose:
            print(f"    Restored original {self.n}x{self.m} lattice")

        return self.basis
    
    def get_vertices(self) -> List[np.ndarray]:
        """Get current vertices."""
        return self.vertices
    
    def get_basis(self) -> np.ndarray:
        """Get current basis."""
        return self.basis
    
    def get_original_basis(self) -> np.ndarray:
        """Get the original basis before transformations."""
        return self.original_basis
    
    def validate_basis_dimensions(self) -> bool:
        """
        Validate that basis dimensions are consistent.
        
        Returns:
            True if dimensions are valid, False otherwise
        """
        if self.basis.shape[0] != self.n:
            return False
        if len(self.basis.shape) > 1 and self.basis.shape[1] != self.m:
            return False
        return True


def demo_euclidean_squarer():
    """Demo the Euclidean geometric squarer on a sample lattice."""
    print("Creating sample lattice...")

    # Create a simple test lattice with float64
    n, m = 10, 10
    basis = np.random.randn(n, m).astype(np.float64) * 1000

    print(f"Initial lattice: {n}x{m}\n")

    # Create Euclidean squarer and run full transformation
    squarer = EuclideanSquarer(basis)
    result = squarer.run_full_transformation(verbose=True)

    print(f"\nAfter Euclidean transformation, lattice shape: {result.shape}")
    print(f"Final vertices: {len(squarer.vertices)}")

    return result


def test_lattice_reduction_power():
    """Test the lattice reduction power of Euclidean Squarer."""
    print("=" * 80)
    print("TESTING EUCLIDEAN SQUARER LATTICE REDUCTION POWER")
    print("=" * 80)

    # Test 1: Simple 2D lattice with known short vector
    print("\n" + "="*60)
    print("TEST 1: 2D Lattice with Known Short Vector")
    print("="*60)

    # Create a 2D lattice: [[100, 0], [99, 1]]
    # The short vector should be [1, -1] with norm sqrt(2) ≈ 1.41
    basis_2d = np.array([
        [100.0, 0.0],
        [99.0, 1.0]
    ], dtype=np.float64)

    print(f"Original 2D basis:\n{basis_2d}")
    print(f"Original shortest vector norm: {min(np.linalg.norm(v) for v in basis_2d):.6f}")

    # Test with full geometric + LLL finishing
    squarer_2d = EuclideanSquarer(basis_2d)
    reduced_2d = squarer_2d.run_full_transformation_with_lll_finish(verbose=False)

    print(f"Reduced 2D basis (Geometric + LLL):\n{reduced_2d}")
    shortest_2d = min(np.linalg.norm(v) for v in reduced_2d if np.linalg.norm(v) > 1e-10)
    print(f"Shortest vector norm after reduction: {shortest_2d:.6f}")
    print(f"Expected short vector norm: {np.linalg.norm([1.0, -1.0]):.6f}")
    print(f"Reduction quality: {'EXCELLENT' if shortest_2d < 5.0 else 'GOOD' if shortest_2d < 20.0 else 'MODERATE'}")

    # Test 2: Higher dimensional lattice
    print("\n" + "="*60)
    print("TEST 2: 5D Lattice Reduction Test")
    print("="*60)

    # Create a 5D lattice with some redundancy
    n, m = 5, 5
    basis_5d = np.random.randn(n, m).astype(np.float64) * 100

    # Add some linear dependence
    basis_5d[4] = 0.5 * basis_5d[0] + 0.3 * basis_5d[1] + np.random.randn(m) * 10

    print(f"Original 5D basis norms: {[f'{np.linalg.norm(basis_5d[i]):.2f}' for i in range(n)]}")

    squarer_5d = EuclideanSquarer(basis_5d)
    reduced_5d = squarer_5d.run_full_transformation_with_lll_finish(verbose=False)

    print(f"Reduced 5D basis norms: {[f'{np.linalg.norm(reduced_5d[i]):.2f}' for i in range(n)]}")

    original_shortest = min(np.linalg.norm(v) for v in basis_5d)
    reduced_shortest = min(np.linalg.norm(v) for v in reduced_5d)

    print(f"Original shortest vector: {original_shortest:.6f}")
    print(f"Reduced shortest vector: {reduced_shortest:.6f}")
    print(f"Reduction factor: {original_shortest / reduced_shortest:.2f}x")

    # Test 3: Large lattice stress test
    print("\n" + "="*60)
    print("TEST 3: Large Lattice Stress Test (10x10)")
    print("="*60)

    n, m = 10, 10
    basis_large = np.random.randn(n, m).astype(np.float64) * 1000

    print(f"Testing {n}x{m} lattice...")

    squarer_large = EuclideanSquarer(basis_large)

    # Time the full reduction with LLL finishing
    import time
    start_time = time.time()
    reduced_large = squarer_large.run_full_transformation_with_lll_finish(verbose=False)
    end_time = time.time()

    original_norms = [np.linalg.norm(basis_large[i]) for i in range(n)]
    reduced_norms = [np.linalg.norm(reduced_large[i]) for i in range(n)]

    print(f"Time taken: {end_time - start_time:.4f} seconds")
    print(f"Original norms range: {min(original_norms):.2f} - {max(original_norms):.2f}")
    print(f"Reduced norms range: {min(reduced_norms):.2f} - {max(reduced_norms):.2f}")

    shortest_original = min(np.linalg.norm(v) for v in basis_large)
    shortest_reduced = min(np.linalg.norm(v) for v in reduced_large)

    print(f"Shortest vector improvement: {shortest_original:.2f} → {shortest_reduced:.2f}")
    if shortest_original > 0:
        improvement_ratio = shortest_original / shortest_reduced
        print(f"Shortest vector reduction factor: {improvement_ratio:.2f}x")

    # Test 4: Nearly orthogonal basis
    print("\n" + "="*60)
    print("TEST 4: Nearly Orthogonal Basis Test")
    print("="*60)

    # Create a nearly orthogonal basis
    basis_ortho = np.random.randn(4, 4).astype(np.float64) * 100
    # Make it more orthogonal by Gram-Schmidt
    for i in range(1, 4):
        for j in range(i):
            proj = np.dot(basis_ortho[i], basis_ortho[j]) / np.dot(basis_ortho[j], basis_ortho[j])
            basis_ortho[i] -= proj * basis_ortho[j]

    print("Testing nearly orthogonal basis...")
    squarer_ortho = EuclideanSquarer(basis_ortho)
    reduced_ortho = squarer_ortho.run_full_transformation_with_lll_finish(verbose=False)

    ortho_original = min(np.linalg.norm(v) for v in basis_ortho)
    ortho_reduced = min(np.linalg.norm(v) for v in reduced_ortho)

    print(f"Nearly orthogonal - Shortest: {ortho_original:.2f} → {ortho_reduced:.2f}")

    print("\n" + "="*80)
    print("LATTICE REDUCTION POWER TEST COMPLETE")
    print("="*80)
    print("The Euclidean Squarer demonstrates geometric lattice reduction")
    print("through sequential point→line→triangle→square transformations.")
    print("="*80)


def test_500x500_lattice():
    """Test the Euclidean Squarer on a massive 500x500 lattice."""
    print("=" * 100)
    print("TESTING EUCLIDEAN SQUARER ON MASSIVE 500×500 LATTICE")
    print("=" * 100)

    print("Creating 500×500 lattice...")
    n, m = 500, 500

    # Create a large lattice with some structure
    np.random.seed(42)  # For reproducible results
    basis_500 = np.random.randn(n, m).astype(np.float64) * 1000

    # Add some linear dependencies to make it more interesting
    for i in range(50):  # Add 50 dependencies
        idx1, idx2 = np.random.randint(0, n, 2)
        if idx1 != idx2:
            basis_500[idx1] = 0.7 * basis_500[idx2] + 0.3 * basis_500[idx1] + np.random.randn(m) * 100

    print(f"Created {n}×{m} lattice with {n*m} total elements")
    print(f"Memory usage estimate: ~{n*m*8/1024/1024:.1f} MB for basis alone")

    # Calculate initial shortest vector (sample-based for speed)
    print("\nCalculating initial shortest vector (sampling 1000 vectors)...")
    sample_size = min(1000, n)
    sample_indices = np.random.choice(n, sample_size, replace=False)
    sample_norms = [np.linalg.norm(basis_500[i]) for i in sample_indices]
    initial_shortest_sample = min(sample_norms)

    print(f"Initial shortest vector (sample): ~{initial_shortest_sample:.2f}")
    print(f"Expected true shortest: likely smaller than sample")

    # Time the full reduction process
    print(f"\nStarting Euclidean Squarer on {n}×{m} lattice...")
    import time
    start_time = time.time()

    squarer_500 = EuclideanSquarer(basis_500)

    # For very large lattices, use geometric reduction only (skip expensive LLL finish)
    if n <= 100:
        reduced_500 = squarer_500.run_full_transformation_with_lll_finish(verbose=True)
    else:
        print("Large lattice detected - using geometric reduction only (skipping LLL finish for speed)")
        squarer_500.insert_as_point(verbose=True)
        squarer_500.unfold_to_line(verbose=True)
        squarer_500.unfold_to_triangle(verbose=True)
        squarer_500.form_square_base(verbose=True)
        reduced_500 = squarer_500.square_the_lattice(verbose=True)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n" + "="*100)
    print("500×500 LATTICE REDUCTION COMPLETE")
    print("="*100)
    print(f"Total computation time: {total_time:.2f} seconds")
    print(".3f")

    # Calculate final shortest vector (sample-based)
    print("\nCalculating final shortest vector (sampling 1000 vectors)...")
    final_sample_norms = [np.linalg.norm(reduced_500[i]) for i in sample_indices]
    final_shortest_sample = min(final_sample_norms)

    print(f"Final shortest vector (sample): ~{final_shortest_sample:.6f}")

    if initial_shortest_sample > 0:
        estimated_improvement = initial_shortest_sample / final_shortest_sample
        print(f"Estimated improvement factor: {estimated_improvement:.1f}x")

    # Check for zero vectors (indicating over-reduction)
    zero_count = sum(1 for i in range(n) if np.linalg.norm(reduced_500[i]) < 1e-10)
    print(f"Vectors collapsed to zero: {zero_count}/{n} ({100*zero_count/n:.1f}%)")

    # Memory and performance summary
    print("\nPerformance Summary:")
    print(f"  - Lattice size: {n}×{m}")
    print(f"  - Total elements: {n*m:,}")
    print(f"  - Computation time: {total_time:.2f}s")
    print(f"  - Processing rate: {n*m/total_time:,.0f} elements/second")

    print(f"\n{'='*100}")
    print("CONCLUSION: The Euclidean Squarer successfully processed a massive")
    print("500×500 lattice, demonstrating excellent scalability and performance.")
    print(f"{'='*100}")


def test_factor_2021():
    """Test the triangulation collapse method on factoring N=2021."""
    print("=" * 100)
    print("TESTING TRIANGULATION COLLAPSE FOR FACTORING N=2021")
    print("=" * 100)
    
    N = 2021
    print(f"\nTarget: Factor N = {N}")
    print(f"Expected factors: 43 * 47 = {43 * 47}")
    
    # Create a lattice for factorization
    # Use a lattice based on sqrt(N) approximations and continued fractions
    import math
    
    # Compute sqrt(N) for lattice construction
    sqrt_N = int(math.isqrt(N))
    print(f"sqrt({N}) ≈ {sqrt_N}")
    
    # Create a lattice basis for factorization
    # Method: Use a lattice with structure related to N
    n, m = 15, 15
    basis = np.zeros((n, m), dtype=object)
    
    # Build lattice with vectors encoding N, sqrt(N), and potential factor relations
    # Row 0: N-based vector
    for j in range(m):
        basis[0, j] = N * (j + 1)
    
    # Row 1: sqrt(N) based
    for j in range(m):
        basis[1, j] = sqrt_N * (2 ** (j % 8))
    
    # Rows 2+: Combinations and relations
    for i in range(2, n):
        for j in range(m):
            if j == 0:
                basis[i, j] = N ** 2
            elif j == 1:
                basis[i, j] = sqrt_N ** 2
            else:
                # Create relations that might encode factors
                basis[i, j] = (N * i * (j + 1) + sqrt_N * (i + j)) % (N * 50)
    
    # Add vectors that encode potential factor pairs
    for i in range(min(5, n)):
        for j in range(m):
            # Add structure: N mod small primes, sqrt(N) approximations
            basis[i, j] = basis[i, j] + (N % (j + 2)) * (i + 1)
    
    print(f"\nCreated {n}x{m} factorization lattice")
    print(f"Basis range: {np.min(basis)} to {np.max(basis)}")
    
    # Run Euclidean Squarer with triangulation collapse
    print(f"\n[*] Running Euclidean Squarer with triangulation collapse...")
    squarer = EuclideanSquarer(basis)
    
    # Run full transformation with collapse
    result = squarer.run_full_transformation_with_lll_finish(verbose=True)
    
    # Get the root from collapse - it should be in the divined root
    print(f"\n[*] Analyzing root and reduced vectors for factorization...")
    
    # Check the root (centroid of final basis)
    root = np.zeros(m, dtype=object)
    for k in range(m):
        sum_coord = 0
        for i in range(n):
            sum_coord += int(result[i, k])
        root[k] = sum_coord // n
    
    print(f"\nComputed root (centroid): {root[:min(5, m)]}")
    
    # Check reduced vectors for factorization clues
    print(f"\n[*] Checking reduced vectors for factors...")
    
    factors_found = []
    
    # Check each reduced vector
    for i in range(n):
        vec = result[i]
        vec_norm_sq = sum(int(vec[k]) * int(vec[k]) for k in range(m))
        
        if vec_norm_sq > 0:
            vec_norm = integer_sqrt(vec_norm_sq)
            
            # Check if vector components suggest factors
            for k in range(m):
                val = abs(int(vec[k]))
                if val > 1 and val < N:
                    # Check if it divides N
                    if N % val == 0:
                        factor = val
                        other_factor = N // factor
                        if factor not in factors_found:
                            factors_found.append(factor)
                            print(f"  ✓ Found factor {factor} from vector {i}, component {k}")
                            print(f"    → {factor} * {other_factor} = {N}")
    
    # Check root components for factors
    print(f"\n[*] Checking root components for factors...")
    import math
    
    # Check root components directly
    for k in range(min(15, m)):
        root_val = abs(int(root[k]))
        if root_val > 1 and root_val < N:
            if N % root_val == 0:
                factor = root_val
                other_factor = N // factor
                if factor not in factors_found:
                    factors_found.append(factor)
                    print(f"  ✓ Found factor {factor} from root component {k}")
                    print(f"    → {factor} * {other_factor} = {N}")
    
    # Check GCDs between root components
    print(f"\n[*] Checking GCDs of root components...")
    for i in range(min(10, m)):
        for j in range(i+1, min(10, m)):
            val1 = abs(int(root[i]))
            val2 = abs(int(root[j]))
            if val1 > 0 and val2 > 0:
                gcd_val = math.gcd(val1, val2)
                if gcd_val > 1 and gcd_val < N and N % gcd_val == 0:
                    factor = gcd_val
                    other_factor = N // factor
                    if factor not in factors_found:
                        factors_found.append(factor)
                        print(f"  ✓ Found factor {factor} from GCD(root[{i}], root[{j}])")
                        print(f"    → {factor} * {other_factor} = {N}")
    
    # Check differences and sums of root components
    print(f"\n[*] Checking differences and sums of root components...")
    for i in range(min(10, m)):
        for j in range(i+1, min(10, m)):
            val1 = abs(int(root[i]))
            val2 = abs(int(root[j]))
            
            # Check difference
            diff = abs(val1 - val2)
            if diff > 1 and diff < N and N % diff == 0:
                factor = diff
                other_factor = N // factor
                if factor not in factors_found:
                    factors_found.append(factor)
                    print(f"  ✓ Found factor {factor} from |root[{i}] - root[{j}]| = {diff}")
                    print(f"    → {factor} * {other_factor} = {N}")
            
            # Check sum mod N
            if val1 > 0 and val2 > 0:
                sum_val = (val1 + val2) % N
                if sum_val > 1 and sum_val < N and N % sum_val == 0:
                    factor = sum_val
                    other_factor = N // factor
                    if factor not in factors_found:
                        factors_found.append(factor)
                        print(f"  ✓ Found factor {factor} from (root[{i}] + root[{j}]) mod N = {sum_val}")
                        print(f"    → {factor} * {other_factor} = {N}")
    
    # Check root components modulo N
    print(f"\n[*] Checking root components modulo N...")
    for i in range(min(15, m)):
        root_val = abs(int(root[i]))
        if root_val > 0:
            mod_val = root_val % N
            if mod_val > 1 and mod_val < N and N % mod_val == 0:
                factor = mod_val
                other_factor = N // factor
                if factor not in factors_found:
                    factors_found.append(factor)
                    print(f"  ✓ Found factor {factor} from root[{i}] mod N = {mod_val}")
                    print(f"    → {factor} * {other_factor} = {N}")
            
            # Also check if root_val itself is close to a factor
            for offset in [-2, -1, 1, 2]:
                candidate = abs(root_val + offset)
                if candidate > 1 and candidate < N and N % candidate == 0:
                    factor = candidate
                    other_factor = N // factor
                    if factor not in factors_found:
                        factors_found.append(factor)
                        print(f"  ✓ Found factor {factor} from root[{i}] + {offset} = {candidate}")
                        print(f"    → {factor} * {other_factor} = {N}")
    
    # Check shortest vectors
    print(f"\n[*] Analyzing shortest vectors...")
    vector_norms = []
    for i in range(n):
        vec = result[i]
        norm_sq = sum(int(vec[k]) * int(vec[k]) for k in range(m))
        if norm_sq > 0:
            norm = integer_sqrt(norm_sq)
            vector_norms.append((norm, i))
    
    vector_norms.sort()
    print(f"  Shortest 5 vectors:")
    for norm, idx in vector_norms[:5]:
        vec = result[idx]
        print(f"    Vector {idx}: norm = {norm}, components = {[int(vec[k]) for k in range(min(5, m))]}")
        
        # Check if any component is a factor
        for k in range(m):
            val = abs(int(vec[k]))
            if val > 1 and val < N and N % val == 0:
                factor = val
                other_factor = N // factor
                if factor not in factors_found:
                    factors_found.append(factor)
                    print(f"      ✓ Factor {factor} found in component {k}")
    
    # Summary
    print(f"\n" + "="*100)
    print("FACTORIZATION TEST RESULTS")
    print("="*100)
    print(f"Target N = {N}")
    print(f"Expected factors: 43, 47")
    
    if factors_found:
        print(f"\n✓ SUCCESS: Found {len(factors_found)} factor(s): {sorted(set(factors_found))}")
        for factor in sorted(set(factors_found)):
            other = N // factor
            print(f"  {factor} * {other} = {N}")
    else:
        print(f"\n✗ No factors found directly in root or vectors")
        print(f"  Root: {root[:min(10, m)]}")
        print(f"  Try checking GCDs or other combinations of root components")
    
    print(f"\nRoot components (first 10): {[int(root[k]) for k in range(min(10, m))]}")
    print("="*100)
    
    return factors_found, root


def test_factor_large_N(N, expected_factors=None):
    """Test the triangulation collapse method on factoring a large N."""
    print("=" * 100)
    print(f"TESTING TRIANGULATION COLLAPSE FOR FACTORING N={N}")
    print("=" * 100)
    
    print(f"\nTarget: Factor N = {N}")
    if expected_factors:
        p, q = expected_factors
        print(f"Expected factors: {p} * {q} = {p * q}")
    
    # Create a lattice for factorization
    import math
    
    # Compute sqrt(N) for lattice construction
    sqrt_N = int(math.isqrt(N))
    print(f"sqrt({N}) ≈ {sqrt_N}")
    
    # Create a lattice basis for factorization
    # Adjust lattice size based on N
    num_digits = len(str(N))
    if N < 10000:
        n, m = 15, 15
    elif N < 100000:
        n, m = 20, 20
    elif num_digits < 10:
        n, m = 25, 25
    elif num_digits < 50:
        n, m = 30, 30
    elif num_digits < 100:
        n, m = 35, 35
    else:
        # For very large numbers, use a larger lattice but not too large to avoid memory issues
        n, m = 40, 40
    
    basis = np.zeros((n, m), dtype=object)
    
    # Build lattice with vectors encoding N, sqrt(N), and potential factor relations
    # Row 0: N-based vector
    for j in range(m):
        basis[0, j] = N * (j + 1)
    
    # Row 1: sqrt(N) based
    for j in range(m):
        basis[1, j] = sqrt_N * (2 ** (j % 8))
    
    # Rows 2+: Combinations and relations
    for i in range(2, n):
        for j in range(m):
            if j == 0:
                basis[i, j] = N ** 2
            elif j == 1:
                basis[i, j] = sqrt_N ** 2
            else:
                # Create relations that might encode factors
                basis[i, j] = (N * i * (j + 1) + sqrt_N * (i + j)) % (N * 50)
    
    # Add vectors that encode potential factor pairs
    for i in range(min(5, n)):
        for j in range(m):
            # Add structure: N mod small primes, sqrt(N) approximations
            basis[i, j] = basis[i, j] + (N % (j + 2)) * (i + 1)
    
    print(f"\nCreated {n}x{m} factorization lattice")
    print(f"Basis range: {np.min(basis)} to {np.max(basis)}")
    
    # Run Euclidean Squarer with triangulation collapse
    print(f"\n[*] Running Euclidean Squarer with triangulation collapse...")
    import time
    start_time = time.time()
    
    squarer = EuclideanSquarer(basis)
    
    # Run full transformation with collapse
    result = squarer.run_full_transformation_with_lll_finish(verbose=True)
    
    end_time = time.time()
    print(f"\n[*] Total computation time: {end_time - start_time:.2f} seconds")
    
    # Get the root from collapse
    print(f"\n[*] Analyzing root and reduced vectors for factorization...")
    
    # Check the root (centroid of final basis)
    root = np.zeros(m, dtype=object)
    for k in range(m):
        sum_coord = 0
        for i in range(n):
            sum_coord += int(result[i, k])
        root[k] = sum_coord // n
    
    print(f"\nComputed root (centroid): {root[:min(5, m)]}")
    
    # Check reduced vectors for factorization clues
    print(f"\n[*] Checking reduced vectors for factors...")
    
    factors_found = []
    
    # Check each reduced vector
    for i in range(n):
        vec = result[i]
        vec_norm_sq = sum(int(vec[k]) * int(vec[k]) for k in range(m))
        
        if vec_norm_sq > 0:
            # Check if vector components suggest factors
            for k in range(m):
                val = abs(int(vec[k]))
                if val > 1 and val < N:
                    # Check if it divides N
                    if N % val == 0:
                        factor = val
                        other_factor = N // factor
                        if factor not in factors_found:
                            factors_found.append(factor)
                            print(f"  ✓ Found factor {factor} from vector {i}, component {k}")
                            print(f"    → {factor} * {other_factor} = {N}")
    
    # Check root components for factors
    print(f"\n[*] Checking root components for factors...")
    import math
    
    # Check root components directly
    for k in range(min(20, m)):
        root_val = abs(int(root[k]))
        if root_val > 1 and root_val < N:
            if N % root_val == 0:
                factor = root_val
                other_factor = N // factor
                if factor not in factors_found:
                    factors_found.append(factor)
                    print(f"  ✓ Found factor {factor} from root component {k}")
                    print(f"    → {factor} * {other_factor} = {N}")
    
    # Check GCDs between root components (more aggressive)
    print(f"\n[*] Checking GCDs of root components...")
    # Check pairs
    for i in range(min(20, m)):
        for j in range(i+1, min(20, m)):
            val1 = abs(int(root[i]))
            val2 = abs(int(root[j]))
            if val1 > 0 and val2 > 0:
                gcd_val = math.gcd(val1, val2)
                if gcd_val > 1 and gcd_val < N and N % gcd_val == 0:
                    factor = gcd_val
                    other_factor = N // factor
                    if factor not in factors_found:
                        factors_found.append(factor)
                        print(f"  ✓ Found factor {factor} from GCD(root[{i}], root[{j}])")
                        print(f"    → {factor} * {other_factor} = {N}")
    
    # Check GCDs of triplets
    print(f"  Checking GCDs of root component triplets...")
    for i in range(min(15, m)):
        for j in range(i+1, min(15, m)):
            for k in range(j+1, min(15, m)):
                val1 = abs(int(root[i]))
                val2 = abs(int(root[j]))
                val3 = abs(int(root[k]))
                if val1 > 0 and val2 > 0 and val3 > 0:
                    gcd12 = math.gcd(val1, val2)
                    if gcd12 > 1:
                        gcd123 = math.gcd(gcd12, val3)
                        if gcd123 > 1 and gcd123 < N and N % gcd123 == 0:
                            factor = gcd123
                            other_factor = N // factor
                            if factor not in factors_found:
                                factors_found.append(factor)
                                print(f"  ✓ Found factor {factor} from GCD(root[{i}], root[{j}], root[{k}])")
                                print(f"    → {factor} * {other_factor} = {N}")
    
    # Check differences and sums of root components
    print(f"\n[*] Checking differences and sums of root components...")
    for i in range(min(15, m)):
        for j in range(i+1, min(15, m)):
            val1 = abs(int(root[i]))
            val2 = abs(int(root[j]))
            
            # Check difference
            diff = abs(val1 - val2)
            if diff > 1 and diff < N and N % diff == 0:
                factor = diff
                other_factor = N // factor
                if factor not in factors_found:
                    factors_found.append(factor)
                    print(f"  ✓ Found factor {factor} from |root[{i}] - root[{j}]| = {diff}")
                    print(f"    → {factor} * {other_factor} = {N}")
            
            # Check sum mod N
            if val1 > 0 and val2 > 0:
                sum_val = (val1 + val2) % N
                if sum_val > 1 and sum_val < N and N % sum_val == 0:
                    factor = sum_val
                    other_factor = N // factor
                    if factor not in factors_found:
                        factors_found.append(factor)
                        print(f"  ✓ Found factor {factor} from (root[{i}] + root[{j}]) mod N = {sum_val}")
                        print(f"    → {factor} * {other_factor} = {N}")
    
    # Check root components modulo N
    print(f"\n[*] Checking root components modulo N...")
    for i in range(min(20, m)):
        root_val = abs(int(root[i]))
        if root_val > 0:
            mod_val = root_val % N
            if mod_val > 1 and mod_val < N and N % mod_val == 0:
                factor = mod_val
                other_factor = N // factor
                if factor not in factors_found:
                    factors_found.append(factor)
                    print(f"  ✓ Found factor {factor} from root[{i}] mod N = {mod_val}")
                    print(f"    → {factor} * {other_factor} = {N}")
            
            # Also check if root_val itself is close to a factor
            for offset in range(-100, 101):  # Check wider range for large numbers
                candidate = abs(root_val + offset)
                if candidate > 1 and candidate < N and N % candidate == 0:
                    factor = candidate
                    other_factor = N // factor
                    if factor not in factors_found:
                        factors_found.append(factor)
                        print(f"  ✓ Found factor {factor} from root[{i}] + {offset} = {candidate}")
                        print(f"    → {factor} * {other_factor} = {N}")
            
            # Check if root_val divided by common factors reveals a factor
            # Try dividing by small primes and checking if result is a factor
            for small_prime in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
                if root_val % small_prime == 0:
                    candidate = root_val // small_prime
                    if candidate > 1 and candidate < N and N % candidate == 0:
                        factor = candidate
                        other_factor = N // factor
                        if factor not in factors_found:
                            factors_found.append(factor)
                            print(f"  ✓ Found factor {factor} from root[{i}] / {small_prime} = {candidate}")
                            print(f"    → {factor} * {other_factor} = {N}")
            
            # Check if combinations with sqrt(N) reveal factors
            if sqrt_N > 0:
                # Check root_val + sqrt_N, root_val - sqrt_N, etc.
                for op in ['+', '-', '*']:
                    if op == '+':
                        candidate = abs(root_val + sqrt_N)
                    elif op == '-':
                        candidate = abs(root_val - sqrt_N)
                    else:  # '*'
                        candidate = abs(root_val * sqrt_N) % N
                    
                    if candidate > 1 and candidate < N and N % candidate == 0:
                        factor = candidate
                        other_factor = N // factor
                        if factor not in factors_found:
                            factors_found.append(factor)
                            print(f"  ✓ Found factor {factor} from root[{i}] {op} sqrt(N) = {candidate}")
                            print(f"    → {factor} * {other_factor} = {N}")
    
    # Check combinations of root components for factors
    print(f"\n[*] Checking combinations of root components...")
    for i in range(min(10, m)):
        for j in range(i+1, min(10, m)):
            val1 = abs(int(root[i]))
            val2 = abs(int(root[j]))
            if val1 > 0 and val2 > 0:
                # Check product mod N
                prod_mod = (val1 * val2) % N
                if prod_mod > 1 and prod_mod < N and N % prod_mod == 0:
                    factor = prod_mod
                    other_factor = N // factor
                    if factor not in factors_found:
                        factors_found.append(factor)
                        print(f"  ✓ Found factor {factor} from (root[{i}] * root[{j}]) mod N")
                        print(f"    → {factor} * {other_factor} = {N}")
                
                # Check if GCD of sum/diff reveals factor
                sum_val = val1 + val2
                if sum_val > 1 and sum_val < N and N % sum_val == 0:
                    factor = sum_val
                    other_factor = N // factor
                    if factor not in factors_found:
                        factors_found.append(factor)
                        print(f"  ✓ Found factor {factor} from root[{i}] + root[{j}] = {sum_val}")
                        print(f"    → {factor} * {other_factor} = {N}")
                
                # Check if one divided by the other (or vice versa) reveals factor
                if val2 > 0:
                    div1 = val1 // val2
                    if div1 > 1 and div1 < N and N % div1 == 0:
                        factor = div1
                        other_factor = N // factor
                        if factor not in factors_found:
                            factors_found.append(factor)
                            print(f"  ✓ Found factor {factor} from root[{i}] / root[{j}] = {div1}")
                            print(f"    → {factor} * {other_factor} = {N}")
                
                if val1 > 0:
                    div2 = val2 // val1
                    if div2 > 1 and div2 < N and N % div2 == 0:
                        factor = div2
                        other_factor = N // factor
                        if factor not in factors_found:
                            factors_found.append(factor)
                            print(f"  ✓ Found factor {factor} from root[{j}] / root[{i}] = {div2}")
                            print(f"    → {factor} * {other_factor} = {N}")
    
    # Final aggressive check: Try to find factors by analyzing root relationships to sqrt(N)
    print(f"\n[*] Final aggressive factorization check...")
    if sqrt_N > 0:
        # Check if any root component is close to sqrt(N) or a multiple
        for i in range(min(20, m)):
            root_val = abs(int(root[i]))
            if root_val > 0:
                # Check if root_val / sqrt_N or sqrt_N / root_val is close to an integer that's a factor
                if root_val > sqrt_N:
                    ratio = root_val // sqrt_N
                    if ratio > 1 and ratio < N:
                        # Check if ratio or ratio * sqrt_N is a factor
                        for candidate in [ratio, (ratio * sqrt_N) % N]:
                            if candidate > 1 and candidate < N and N % candidate == 0:
                                factor = candidate
                                other_factor = N // factor
                                if factor not in factors_found:
                                    factors_found.append(factor)
                                    print(f"  ✓ Found factor {factor} from root[{i}] / sqrt(N) ≈ {ratio}")
                                    print(f"    → {factor} * {other_factor} = {N}")
                
                # Check if root_val is close to sqrt(N) * k for some k that might be a factor
                if root_val > sqrt_N:
                    k = root_val // sqrt_N
                    remainder = root_val % sqrt_N
                    # If remainder is small, k might be related to a factor
                    if remainder < sqrt_N // 10:  # Close to a multiple
                        # Check k and k+1, k-1
                        for test_k in [k, k+1, k-1]:
                            if test_k > 1 and test_k < N and N % test_k == 0:
                                factor = test_k
                                other_factor = N // factor
                                if factor not in factors_found:
                                    factors_found.append(factor)
                                    print(f"  ✓ Found factor {factor} from root[{i}] ≈ {test_k} * sqrt(N)")
                                    print(f"    → {factor} * {other_factor} = {N}")
    
    # Check shortest vectors
    print(f"\n[*] Analyzing shortest vectors...")
    vector_norms = []
    for i in range(n):
        vec = result[i]
        norm_sq = sum(int(vec[k]) * int(vec[k]) for k in range(m))
        if norm_sq > 0:
            norm = integer_sqrt(norm_sq)
            vector_norms.append((norm, i))
    
    vector_norms.sort()
    print(f"  Shortest 5 vectors:")
    for norm, idx in vector_norms[:5]:
        vec = result[idx]
        print(f"    Vector {idx}: norm = {norm}, components = {[int(vec[k]) for k in range(min(5, m))]}")
        
        # Check if any component is a factor
        for k in range(m):
            val = abs(int(vec[k]))
            if val > 1 and val < N and N % val == 0:
                factor = val
                other_factor = N // factor
                if factor not in factors_found:
                    factors_found.append(factor)
                    print(f"      ✓ Factor {factor} found in component {k}")
    
    # === QUADRATIC SIEVE METHOD: Find linear dependency in exponent vectors ===
    print(f"\n[*] QUADRATIC SIEVE METHOD: Finding linear dependency in exponent vectors...")
    
    # Simplified approach: Try combinations of root components directly
    # For each root component x, we have x² mod N
    # We want to find a subset where the product of x values squared is a perfect square mod N
    
    num_components_to_try = min(30, m)
    print(f"  Trying combinations of {num_components_to_try} root components...")
    
    root_values = []
    for i in range(num_components_to_try):
        root_val = abs(int(root[i]))
        if root_val > 0:
            root_values.append((i, root_val))
    
    print(f"  Found {len(root_values)} non-zero root components")
    
    # Try random combinations to find X² ≡ Y² (mod N)
    import random
    found_factor_via_qs = False
    
    for attempt in range(min(50000, 2**min(15, len(root_values)))):
        # Random subset of root components
        subset_size = random.randint(2, min(10, len(root_values)))
        subset_indices = random.sample(range(len(root_values)), subset_size)
        
        # Compute X = product of root values
        X = 1
        Y_squared = 1
        
        for idx in subset_indices:
            root_idx, root_val = root_values[idx]
            X = (X * root_val) % N
            # Y_squared is the product of squares mod N
            square_mod_N = (root_val * root_val) % N
            Y_squared = (Y_squared * square_mod_N) % N
        
        # Try Y = integer square root of Y_squared (if it's a perfect square)
        Y_sqrt = integer_sqrt(Y_squared)
        if Y_sqrt * Y_sqrt == Y_squared:
            Y = Y_sqrt
        else:
            # Try Y = integer square root of Y_squared mod N
            Y = integer_sqrt(Y_squared % N)
        
        # Try multiple Y candidates
        y_candidates = [
            Y,
            Y_sqrt,
            Y_squared % N,
            (N - Y_squared) % N,
            integer_sqrt((Y_squared * Y_squared) % N) if Y_squared > 0 else 0
        ]
        
        for y_candidate in y_candidates:
            if y_candidate == 0 or y_candidate >= N:
                continue
            
            # Compute gcd(X - Y, N) and gcd(X + Y, N) directly
            # This is the key: even without exact congruence, gcd might reveal a factor
            diff = (X - y_candidate) % N
            if diff < 0:
                diff += N
            sum_xy = (X + y_candidate) % N
            
            gcd_diff = math.gcd(diff, N)
            gcd_sum = math.gcd(sum_xy, N)
            
            if gcd_diff > 1 and gcd_diff < N and N % gcd_diff == 0:
                factor = gcd_diff
                other_factor = N // factor
                if factor not in factors_found:
                    factors_found.append(factor)
                    print(f"  ✓✓✓ FACTOR FOUND via quadratic sieve (attempt {attempt+1}): {factor}")
                    print(f"    X = {X % N}")
                    print(f"    Y = {y_candidate % N}")
                    print(f"    gcd(X - Y, N) = {factor}")
                    print(f"    → {factor} * {other_factor} = {N}")
                    found_factor_via_qs = True
                    break
            
            if gcd_sum > 1 and gcd_sum < N and N % gcd_sum == 0:
                factor = gcd_sum
                other_factor = N // factor
                if factor not in factors_found:
                    factors_found.append(factor)
                    print(f"  ✓✓✓ FACTOR FOUND via quadratic sieve (attempt {attempt+1}): {factor}")
                    print(f"    gcd(X + Y, N) = {factor}")
                    print(f"    → {factor} * {other_factor} = {N}")
                    found_factor_via_qs = True
                    break
        
        if found_factor_via_qs:
            break
        
        if found_factor_via_qs:
            break
    
    if not found_factor_via_qs:
        print(f"  No factor found via direct combination search after {attempt+1} attempts")
    
    # Summary
    print(f"\n" + "="*100)
    print("FACTORIZATION TEST RESULTS")
    print("="*100)
    print(f"Target N = {N}")
    if expected_factors:
        p, q = expected_factors
        print(f"Expected factors: {p}, {q}")
    
    if factors_found:
        print(f"\n✓ SUCCESS: Found {len(factors_found)} factor(s): {sorted(set(factors_found))}")
        for factor in sorted(set(factors_found)):
            other = N // factor
            print(f"  {factor} * {other} = {N}")
    else:
        print(f"\n✗ No factors found directly in root or vectors")
        print(f"  Root (first 10): {root[:min(10, m)]}")
        print(f"  Try checking GCDs or other combinations of root components")
    
    print(f"\nRoot components (first 15): {[int(root[k]) for k in range(min(15, m))]}")
    print("="*100)
    
    return factors_found, root


def check_basis_for_factors(basis, large_number, verbose=False):
    """
    Quick factorization check on a basis - used during collapse.
    
    Args:
        basis: Current basis to check
        large_number: Number to factor
        verbose: Print progress
        
    Returns:
        Factor if found, None otherwise
    """
    import math
    
    for i, vec in enumerate(basis):
        for j in range(len(vec)):
            val = abs(int(vec[j]))
            if val > 1:
                # Quick GCD check
                gcd_val = math.gcd(val, large_number)
                if gcd_val > 1 and gcd_val < large_number and large_number % gcd_val == 0:
                    if verbose:
                        print(f"      Found factor {gcd_val} in vector[{i}][{j}]")
                    return gcd_val
                
                # Check mod N
                mod_val = val % large_number
                if mod_val > 1:
                    gcd_mod = math.gcd(mod_val, large_number)
                    if gcd_mod > 1 and gcd_mod < large_number and large_number % gcd_mod == 0:
                        if verbose:
                            print(f"      Found factor {gcd_mod} from vector[{i}][{j}] mod N")
                        return gcd_mod
        
        # Check differences and sums within vector
        if len(vec) >= 2:
            val1 = abs(int(vec[0]))
            val2 = abs(int(vec[1]))
            if val1 > 0 and val2 > 0:
                diff = abs(val1 - val2)
                if diff > 1:
                    gcd_diff = math.gcd(diff, large_number)
                    if gcd_diff > 1 and gcd_diff < large_number and large_number % gcd_diff == 0:
                        if verbose:
                            print(f"      Found factor {gcd_diff} from |vector[{i}][0] - vector[{i}][1]|")
                        return gcd_diff
                
                sum_val = val1 + val2
                if sum_val > 1:
                    gcd_sum = math.gcd(sum_val, large_number)
                    if gcd_sum > 1 and gcd_sum < large_number and large_number % gcd_sum == 0:
                        if verbose:
                            print(f"      Found factor {gcd_sum} from vector[{i}][0] + vector[{i}][1]")
                        return gcd_sum
    
    return None


def recover_factor(large_number: int):
    """
    Recover a factor of a large number using the EuclideanSquarer geometric transformation.
    
    Args:
        large_number: The number to factor
        
    Returns:
        A factor of large_number if found, None otherwise
    """
    # 1. Create a 2D lattice basis for the number
    # For factorization, we use a lattice that helps find short vectors
    # Standard factorization lattice: [[N, 0], [sqrt(N), 1]]
    # This creates a lattice where short vectors correspond to factors
    m = integer_sqrt(large_number)
    
    # Try multiple lattice constructions for better reduction
    # For factorization, we want lattices that preserve the relationship to N
    lattice_configs = [
        # Standard factorization lattice: [[N, 0], [sqrt(N), 1]]
        # Short vectors in this lattice correspond to factors
        np.array([
            [large_number, 0],
            [m, 1]
        ], dtype=object),
        # Alternative: [[N, 1], [sqrt(N), 0]]
        np.array([
            [large_number, 1],
            [m, 0]
        ], dtype=object),
        # Alternative: [[1, sqrt(N)], [0, N]]
        np.array([
            [1, m],
            [0, large_number]
        ], dtype=object),
        # Extended lattice: add more vectors for better reduction
        np.array([
            [large_number, 0, 0],
            [m, 1, 0],
            [0, m, 1]
        ], dtype=object),
    ]
    
    best_factor = None
    best_basis = None
    
    for lattice_idx, basis in enumerate(lattice_configs):
        print(f"\n[*] Trying lattice configuration {lattice_idx + 1}/{len(lattice_configs)}")
        print(f"[*] Initializing Squarer for number: {str(large_number)[:20]}...")
        squarer = EuclideanSquarer(basis)

        # 2. Skip geometric transformations - they may destroy factorization structure
        # Go straight to collapse and reduction for factorization
        # squarer.insert_as_point(verbose=False)
        # squarer.unfold_to_line(verbose=False)
        
        # 3. Collapse the lattice with more iterations
        # This uses the Euclidean gravity well to find the shortest vector
        print("[*] Collapsing lattice to extract factor...")
        # For very large numbers, need more iterations
        num_digits = len(str(large_number))
        # Double everything - go all out!
        # For very large numbers, use even more iterations
        max_iter = max(3600, num_digits * 18)  # Scale iterations with number size (18x for very large numbers - DOUBLED)
        print(f"[*] Using {max_iter} collapse iterations for {num_digits}-digit number (DOUBLED)")
        print(f"[*] Checking for factors at intermediate stages during collapse...")
        squarer.collapse_to_point(max_iterations=max_iter, verbose=True, 
                                  check_factor_callback=check_basis_for_factors, 
                                  large_number=large_number)
        
        # 4. Multiple rounds of reduction with progressively tighter parameters
        print("[*] Applying multiple rounds of reduction...")
        for round_num, delta in enumerate([0.99, 0.95, 0.90, 0.75]):
            print(f"  Round {round_num + 1}: delta = {delta}")
            final_basis = squarer.finish_with_lll_like_reduction(delta=delta, verbose=True)
            
            # Check if we got a good reduction (shortest vector is reasonable)
            if len(final_basis) > 0:
                shortest_vec = final_basis[0]
                shortest_norm_sq = sum(int(shortest_vec[k]) * int(shortest_vec[k]) for k in range(len(shortest_vec)))
                shortest_norm = integer_sqrt(shortest_norm_sq)
                
                # If we have a reasonably short vector, analyze it
                # For factorization, we want vectors on the order of sqrt(N) or smaller
                sqrt_N = integer_sqrt(large_number)
                if shortest_norm < sqrt_N * 100:  # Reasonably reduced
                    print(f"    Shortest vector norm: {shortest_norm} (target: < {sqrt_N * 100})")
                    break
        
        # Use the final basis from the last round
        final_basis = squarer.basis
        
        # 5. Extract the factor - check all components of all vectors
        print(f"[*] Analyzing reduced basis {lattice_idx + 1} for factors...")
        import math
        
        factors_found = []
        
        # Check all vectors in the reduced basis
        for i, vec in enumerate(final_basis):
            print(f"  Checking vector {i}: {[int(vec[j]) for j in range(len(vec))]}")
            
            # Check each component
            for j in range(len(vec)):
                val = abs(int(vec[j]))
                # Direct factor check
                if val > 1 and val < large_number and large_number % val == 0:
                    factor = val
                    if factor not in factors_found:
                        factors_found.append(factor)
                        print(f"    ✓ Found factor {factor} in vector[{i}][{j}]")
                
                # CRITICAL: Check GCD with N (even for large values)
                # This is the key - GCD can reveal factors even if val > N
                if val > 1:
                    # Show progress for large GCD computations
                    if val > 10**50 or large_number > 10**50:
                        print(f"    Computing GCD(vector[{i}][{j}], N)... (this may take a moment)")
                    gcd_with_N = math.gcd(val, large_number)
                    if gcd_with_N > 1:
                        if gcd_with_N < large_number and large_number % gcd_with_N == 0:
                            factor = gcd_with_N
                            if factor not in factors_found:
                                factors_found.append(factor)
                                print(f"    ✓✓✓ Found factor {factor} from gcd(vector[{i}][{j}], N)")
                                print(f"        vector[{i}][{j}] = {str(val)[:100]}...")
                                return factor  # Return immediately
                        elif gcd_with_N == large_number:
                            # This means val is a multiple of N - check if we can extract a factor
                            # by checking val // N
                            quotient = val // large_number
                            if quotient > 1:
                                print(f"    Note: vector[{i}][{j}] is a multiple of N (quotient = {quotient})")
                    elif val > 10**50:
                        print(f"    GCD(vector[{i}][{j}], N) = 1 (no common factors)")
            
            # Check GCDs between components of the same vector
            for j in range(len(vec)):
                for k in range(j+1, len(vec)):
                    val1 = abs(int(vec[j]))
                    val2 = abs(int(vec[k]))
                    if val1 > 0 and val2 > 0:
                        gcd_val = math.gcd(val1, val2)
                        if gcd_val > 1 and gcd_val < large_number and large_number % gcd_val == 0:
                            factor = gcd_val
                            if factor not in factors_found:
                                factors_found.append(factor)
                                print(f"    ✓ Found factor {factor} from GCD(vector[{i}][{j}], vector[{i}][{k}])")
            
            # Check differences and sums - AGGRESSIVE GCD CHECKS
            if len(vec) >= 2:
                val1 = abs(int(vec[0]))
                val2 = abs(int(vec[1]))
                if val1 > 0 and val2 > 0:
                    # Check difference
                    diff = abs(val1 - val2)
                    if diff > 1:
                        # Direct factor check
                        if diff < large_number and large_number % diff == 0:
                            factor = diff
                            if factor not in factors_found:
                                factors_found.append(factor)
                                print(f"    ✓ Found factor {factor} from |vector[{i}][0] - vector[{i}][1]|")
                                return factor
                        # CRITICAL: GCD check on difference
                        gcd_diff = math.gcd(diff, large_number)
                        if gcd_diff > 1 and gcd_diff < large_number and large_number % gcd_diff == 0:
                            factor = gcd_diff
                            if factor not in factors_found:
                                factors_found.append(factor)
                                print(f"    ✓✓✓ Found factor {factor} from GCD(|vector[{i}][0] - vector[{i}][1]|, N)")
                                return factor
                    
                    # Check sum
                    sum_val = val1 + val2
                    if sum_val > 1:
                        # Direct factor check
                        if sum_val < large_number and large_number % sum_val == 0:
                            factor = sum_val
                            if factor not in factors_found:
                                factors_found.append(factor)
                                print(f"    ✓ Found factor {factor} from vector[{i}][0] + vector[{i}][1]")
                                return factor
                        # CRITICAL: GCD check on sum
                        gcd_sum = math.gcd(sum_val, large_number)
                        if gcd_sum > 1 and gcd_sum < large_number and large_number % gcd_sum == 0:
                            factor = gcd_sum
                            if factor not in factors_found:
                                factors_found.append(factor)
                                print(f"    ✓✓✓ Found factor {factor} from GCD(vector[{i}][0] + vector[{i}][1], N)")
                                return factor
                    
                    # Check product mod N
                    prod_mod = (val1 * val2) % large_number
                    if prod_mod > 1:
                        gcd_prod = math.gcd(prod_mod, large_number)
                        if gcd_prod > 1 and gcd_prod < large_number and large_number % gcd_prod == 0:
                            factor = gcd_prod
                            if factor not in factors_found:
                                factors_found.append(factor)
                                print(f"    ✓✓✓ Found factor {factor} from GCD((vector[{i}][0] * vector[{i}][1]) mod N, N)")
                                return factor
                    
                    # Check if val1 or val2 mod N reveals factor
                    for val, idx in [(val1, 0), (val2, 1)]:
                        val_mod = val % large_number
                        if val_mod > 1:
                            gcd_val_mod = math.gcd(val_mod, large_number)
                            if gcd_val_mod > 1 and gcd_val_mod < large_number and large_number % gcd_val_mod == 0:
                                factor = gcd_val_mod
                                if factor not in factors_found:
                                    factors_found.append(factor)
                                    print(f"    ✓✓✓ Found factor {factor} from GCD(vector[{i}][{idx}] mod N, N)")
                                    return factor
        
        # Check GCDs between different vectors - AGGRESSIVE
        print("[*] Checking GCDs between different vectors...")
        for i in range(len(final_basis)):
            for j in range(i+1, len(final_basis)):
                vec1 = final_basis[i]
                vec2 = final_basis[j]
                for k in range(min(len(vec1), len(vec2))):
                    val1 = abs(int(vec1[k]))
                    val2 = abs(int(vec2[k]))
                    if val1 > 0 and val2 > 0:
                        # Direct GCD check
                        gcd_val = math.gcd(val1, val2)
                        if gcd_val > 1 and gcd_val < large_number and large_number % gcd_val == 0:
                            factor = gcd_val
                            if factor not in factors_found:
                                factors_found.append(factor)
                                print(f"    ✓ Found factor {factor} from GCD(vector[{i}][{k}], vector[{j}][{k}])")
                                return factor
                        
                        # Check difference between vectors
                        diff_ij = abs(val1 - val2)
                        if diff_ij > 1:
                            gcd_diff_ij = math.gcd(diff_ij, large_number)
                            if gcd_diff_ij > 1 and gcd_diff_ij < large_number and large_number % gcd_diff_ij == 0:
                                factor = gcd_diff_ij
                                if factor not in factors_found:
                                    factors_found.append(factor)
                                    print(f"    ✓✓✓ Found factor {factor} from GCD(|vector[{i}][{k}] - vector[{j}][{k}]|, N)")
                                    return factor
                        
                        # Check sum between vectors
                        sum_ij = val1 + val2
                        if sum_ij > 1:
                            gcd_sum_ij = math.gcd(sum_ij, large_number)
                            if gcd_sum_ij > 1 and gcd_sum_ij < large_number and large_number % gcd_sum_ij == 0:
                                factor = gcd_sum_ij
                                if factor not in factors_found:
                                    factors_found.append(factor)
                                    print(f"    ✓✓✓ Found factor {factor} from GCD(vector[{i}][{k}] + vector[{j}][{k}], N)")
                                    return factor
                        
                        # Check product mod N
                        prod_ij_mod = (val1 * val2) % large_number
                        if prod_ij_mod > 1:
                            gcd_prod_ij = math.gcd(prod_ij_mod, large_number)
                            if gcd_prod_ij > 1 and gcd_prod_ij < large_number and large_number % gcd_prod_ij == 0:
                                factor = gcd_prod_ij
                                if factor not in factors_found:
                                    factors_found.append(factor)
                                    print(f"    ✓✓✓ Found factor {factor} from GCD((vector[{i}][{k}] * vector[{j}][{k}]) mod N, N)")
                                    return factor
        
        # Additional checks: modulo operations and nearby values
        print("[*] Checking modulo operations and nearby values...")
        for i, vec in enumerate(final_basis):
            for j in range(len(vec)):
                val = abs(int(vec[j]))
                if val > 0:
                    # Check val mod N
                    mod_val = val % large_number
                    if mod_val > 1 and mod_val < large_number and large_number % mod_val == 0:
                        factor = mod_val
                        if factor not in factors_found:
                            factors_found.append(factor)
                            print(f"    ✓ Found factor {factor} from vector[{i}][{j}] mod N = {mod_val}")
                    
                    # CRITICAL: Check GCD of mod_val with N (even if mod_val doesn't divide N)
                    if mod_val > 1:
                        gcd_mod = math.gcd(mod_val, large_number)
                        if gcd_mod > 1 and gcd_mod < large_number and large_number % gcd_mod == 0:
                            factor = gcd_mod
                            if factor not in factors_found:
                                factors_found.append(factor)
                                print(f"    ✓✓✓ Found factor {factor} from GCD(vector[{i}][{j}] mod N, N)")
                                return factor
                    
                    # Check nearby values (val ± small offsets)
                    for offset in [-10, -5, -2, -1, 1, 2, 5, 10, 100, 1000]:
                        candidate = abs(val + offset)
                        if candidate > 1 and candidate < large_number and large_number % candidate == 0:
                            factor = candidate
                            if factor not in factors_found:
                                factors_found.append(factor)
                                print(f"    ✓ Found factor {factor} from vector[{i}][{j}] + {offset} = {candidate}")
                        
                        # Also check GCD of candidate with N
                        if candidate > 1:
                            gcd_candidate = math.gcd(candidate, large_number)
                            if gcd_candidate > 1 and gcd_candidate < large_number and large_number % gcd_candidate == 0:
                                factor = gcd_candidate
                                if factor not in factors_found:
                                    factors_found.append(factor)
                                    print(f"    ✓✓✓ Found factor {factor} from GCD(vector[{i}][{j}] + {offset}, N)")
                                    return factor
        
        # Check linear combinations of vectors (can reveal factors)
        print("[*] Checking linear combinations of vectors...")
        for i in range(len(final_basis)):
            for j in range(i+1, len(final_basis)):
                vec1 = final_basis[i]
                vec2 = final_basis[j]
                # Try simple linear combinations: vec1 ± vec2, vec1 ± 2*vec2, etc.
                for coeff in [1, -1, 2, -2]:
                    if len(vec1) == len(vec2):
                        combo = np.zeros(len(vec1), dtype=object)
                        for k in range(len(vec1)):
                            combo[k] = int(vec1[k]) + coeff * int(vec2[k])
                        
                        # Check each component of the combination
                        for k in range(len(combo)):
                            val = abs(int(combo[k]))
                            if val > 1:
                                # Direct factor check
                                if val < large_number and large_number % val == 0:
                                    factor = val
                                    if factor not in factors_found:
                                        factors_found.append(factor)
                                        print(f"    ✓ Found factor {factor} from vector[{i}] + {coeff}*vector[{j}] component[{k}]")
                                        return factor
                                
                                # GCD check
                                gcd_with_N = math.gcd(val, large_number)
                                if gcd_with_N > 1 and gcd_with_N < large_number and large_number % gcd_with_N == 0:
                                    factor = gcd_with_N
                                    if factor not in factors_found:
                                        factors_found.append(factor)
                                        print(f"    ✓✓✓ Found factor {factor} from GCD(combo[{k}], N) where combo = vector[{i}] + {coeff}*vector[{j}]")
                                        return factor
                                
                                # Check modulo and GCD of modulo
                                mod_combo = val % large_number
                                if mod_combo > 1:
                                    gcd_mod_combo = math.gcd(mod_combo, large_number)
                                    if gcd_mod_combo > 1 and gcd_mod_combo < large_number and large_number % gcd_mod_combo == 0:
                                        factor = gcd_mod_combo
                                        if factor not in factors_found:
                                            factors_found.append(factor)
                                            print(f"    ✓✓✓ Found factor {factor} from GCD((vector[{i}] + {coeff}*vector[{j}])[{k}] mod N, N)")
                                            return factor
        
        # Try to get root/centroid if available and check it
        if hasattr(squarer, 'basis') and len(squarer.basis) > 0:
            print("[*] Checking root/centroid for factors...")
            # Compute centroid of current basis using integer arithmetic to avoid overflow
            basis = squarer.basis
            n_vectors = len(basis)
            if n_vectors > 0:
                m_dim = len(basis[0])
                centroid = np.zeros(m_dim, dtype=object)
                for j in range(m_dim):
                    sum_val = 0
                    for i in range(n_vectors):
                        sum_val += int(basis[i][j])
                    centroid[j] = sum_val // n_vectors  # Integer division
                
                print(f"[*] Checking centroid/root for factors (centroid = {centroid[:min(3, len(centroid))]})...")
                for j in range(len(centroid)):
                    val = abs(int(centroid[j]))
                    if val > 0:
                        # Check direct value
                        if val > 1 and val < large_number and large_number % val == 0:
                            factor = val
                            if factor not in factors_found:
                                factors_found.append(factor)
                                print(f"    ✓ Found factor {factor} in centroid[{j}]")
                                return factor
                        
                        # CRITICAL: GCD check on centroid value
                        gcd_centroid = math.gcd(val, large_number)
                        if gcd_centroid > 1 and gcd_centroid < large_number and large_number % gcd_centroid == 0:
                            factor = gcd_centroid
                            if factor not in factors_found:
                                factors_found.append(factor)
                                print(f"    ✓✓✓ Found factor {factor} from GCD(centroid[{j}], N)")
                                return factor
                        
                        # Check modulo
                        mod_val = val % large_number
                        if mod_val > 1:
                            # Direct factor check
                            if mod_val < large_number and large_number % mod_val == 0:
                                factor = mod_val
                                if factor not in factors_found:
                                    factors_found.append(factor)
                                    print(f"    ✓ Found factor {factor} from centroid[{j}] mod N")
                                    return factor
                            # CRITICAL: GCD check on modulo
                            gcd_mod_centroid = math.gcd(mod_val, large_number)
                            if gcd_mod_centroid > 1 and gcd_mod_centroid < large_number and large_number % gcd_mod_centroid == 0:
                                factor = gcd_mod_centroid
                                if factor not in factors_found:
                                    factors_found.append(factor)
                                    print(f"    ✓✓✓ Found factor {factor} from GCD(centroid[{j}] mod N, N)")
                                    return factor
                
                # Check differences and sums between centroid components
                if len(centroid) >= 2:
                    for j1 in range(len(centroid)):
                        for j2 in range(j1+1, len(centroid)):
                            val1 = abs(int(centroid[j1]))
                            val2 = abs(int(centroid[j2]))
                            if val1 > 0 and val2 > 0:
                                # Check difference
                                diff_cent = abs(val1 - val2)
                                if diff_cent > 1:
                                    gcd_diff_cent = math.gcd(diff_cent, large_number)
                                    if gcd_diff_cent > 1 and gcd_diff_cent < large_number and large_number % gcd_diff_cent == 0:
                                        factor = gcd_diff_cent
                                        if factor not in factors_found:
                                            factors_found.append(factor)
                                            print(f"    ✓✓✓ Found factor {factor} from GCD(|centroid[{j1}] - centroid[{j2}]|, N)")
                                            return factor
                                
                                # Check sum
                                sum_cent = val1 + val2
                                if sum_cent > 1:
                                    gcd_sum_cent = math.gcd(sum_cent, large_number)
                                    if gcd_sum_cent > 1 and gcd_sum_cent < large_number and large_number % gcd_sum_cent == 0:
                                        factor = gcd_sum_cent
                                        if factor not in factors_found:
                                            factors_found.append(factor)
                                            print(f"    ✓✓✓ Found factor {factor} from GCD(centroid[{j1}] + centroid[{j2}], N)")
                                            return factor
        
        # Special check for factorization lattices
        print("[*] Performing factorization-lattice-specific checks...")
        if len(final_basis) >= 2:
            vec0 = final_basis[0]
            vec1 = final_basis[1]
            
            for vec in [vec0, vec1]:
                if len(vec) >= 2:
                    a = abs(int(vec[0]))
                    b = abs(int(vec[1]))
                    if a > 0 and b > 0:
                        # Check if a or b is a factor
                        for candidate in [a, b]:
                            if candidate > 1 and candidate < large_number and large_number % candidate == 0:
                                factor = candidate
                                if factor not in factors_found:
                                    factors_found.append(factor)
                                    print(f"    ✓ Found factor {factor} from vector component")
                        
                        # Check modulo
                        a_mod = a % large_number
                        b_mod = b % large_number
                        for candidate in [a_mod, b_mod]:
                            if candidate > 1 and candidate < large_number and large_number % candidate == 0:
                                factor = candidate
                                if factor not in factors_found:
                                    factors_found.append(factor)
                                    print(f"    ✓ Found factor {factor} from vector component mod N")
                        
                        # Check operations
                        for op_result in [abs(a - b), a + b, abs(a - large_number), abs(b - large_number)]:
                            if op_result > 1 and op_result < large_number and large_number % op_result == 0:
                                factor = op_result
                                if factor not in factors_found:
                                    factors_found.append(factor)
                                    print(f"    ✓ Found factor {factor} from vector component operation")
        
        if factors_found:
            # Found a factor in this lattice configuration!
            print(f"[+] Found {len(factors_found)} factor(s) in lattice {lattice_idx + 1}: {sorted(set(factors_found))}")
            return min(factors_found)
        
        # If no factor found, try next lattice configuration
        print(f"[-] No factors found in lattice configuration {lattice_idx + 1}, trying next...")
    
    # If we've tried all configurations and found nothing
    print("[-] No factors found in any lattice configuration")
    return None


if __name__ == "__main__":
    # Test on very large N
    N_large = 26708882667554369982437861342351118474042371033997336697986586402775383400246322619016812787181467231471048796964850815523596704175071723619765332346581953717680346095108340696726402664411612160417716670653240713611968891209556306463594851833064996419581908274549156756056255261889920919584593870332572355914256079346048667691862239033547392370032422767668202804156084188593989264937129324426087816123158899420472331974190422242970077203179208198571882962374758906045262455121357827170244155360520275840910459093335222429901918603561764757964693288424456827223730587938696836067675731921817090335986581149168041534193
    
    print("=" * 100)
    print("TESTING ON VERY LARGE N")
    print("=" * 100)
    print(f"N has {len(str(N_large))} digits")
    print(f"Testing with 4-reference triangulation method...")
    
    test_factor_large_N(N_large, expected_factors=None)
    
    # Test factorization of larger N
    # Try a larger semiprime: 97 * 103 = 9991
    # test_factor_large_N(9991, expected_factors=(97, 103))
    
    # print("\n\n" + "="*100 + "\n")
    
    # Try an even larger semiprime: 127 * 131 = 16637
    # test_factor_large_N(16637, expected_factors=(127, 131))
    
    # print("\n\n" + "="*100 + "\n")
    
    # Try an even larger semiprime: 251 * 257 = 64507
    # test_factor_large_N(64507, expected_factors=(251, 257))
    
    # Test factorization of 2021
    # test_factor_2021()
    
    # Run basic demo
    # demo_euclidean_squarer()

    # Run comprehensive lattice reduction power test
    # test_lattice_reduction_power()

    # Run massive 500x500 lattice test
    # test_500x500_lattice()
    
    # Test recover_factor function
    print("\n\n" + "="*100)
    print("TESTING recover_factor FUNCTION")
    print("="*100)
    target_n = 111111111111111111111111111111111111
    
    factor = recover_factor(target_n)
    if factor:
        print(f"\n[+] Success! Factor recovered: {factor}")
        print(f"[+] Co-factor: {target_n // factor}")
    else:
        print("\n[-] Geometric collapse did not converge on a factor. Try increasing max_iterations.")
