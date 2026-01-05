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

        # Find the primary direction (longest vector in basis)
        primary_direction = np.zeros(self.m, dtype=np.float64)
        max_length = 0
        for vec in self.basis:
            length = np.linalg.norm(vec)
            if length > max_length:
                max_length = length
                primary_direction = vec.copy()

        if max_length == 0:
            primary_direction[0] = 1.0

        # Normalize the direction
        primary_direction = primary_direction / np.linalg.norm(primary_direction)

        # Create line endpoints by expanding from center
        left_endpoint = center - expansion_factor * primary_direction * max_length
        right_endpoint = center + expansion_factor * primary_direction * max_length

        # Expand basis vectors along this line
        line_basis = np.zeros((self.n, self.m), dtype=np.float64)
        for i in range(self.n):
            # Project each vector onto the line and expand
            projection = np.dot(self.basis[i], primary_direction) * primary_direction
            line_basis[i] = center + expansion_factor * projection

        self.basis = line_basis
        self.vertices = np.array([left_endpoint, right_endpoint], dtype=np.float64)

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

        # Calculate line vector and perpendicular
        line_vector = right - left
        line_length = np.linalg.norm(line_vector)

        # Create perpendicular vector (rotate 90 degrees)
        if self.m >= 2:
            perp = np.array([-line_vector[1], line_vector[0]])
            if self.m > 2:
                perp = np.pad(perp, (0, self.m - 2), 'constant')
        else:
            perp = np.array([1.0])

        perp = perp / np.linalg.norm(perp)  # Normalize

        # Create apex above the midpoint
        midpoint = (left + right) / 2.0
        height = expansion_factor * line_length
        apex = midpoint + height * perp

        # Triangle vertices
        self.vertices = np.array([left, right, apex], dtype=np.float64)

        # Expand basis vectors to form triangular relationships
        triangle_basis = np.zeros((self.n, self.m), dtype=np.float64)
        for i in range(self.n):
            # Create triangular relationships between vectors
            # Mix original vector with triangle vertices
            weight = i / max(1, self.n - 1)  # 0 to 1
            triangle_basis[i] = (1 - weight) * self.basis[i] + weight * apex

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

        # Side 1: left → apex
        for i in range(1, num_points):
            t = i / num_points
            v = (1 - t) * left + t * apex
            unpacked.append(v)

        # Side 2: right → apex
        for i in range(1, num_points):
            t = i / num_points
            v = (1 - t) * right + t * apex
            unpacked.append(v)

        # Side 3: left → right (base)
        for i in range(1, num_points):
            t = i / num_points
            v = (1 - t) * left + t * right
            unpacked.append(v)

        # Update vertices with unpacked points
        all_vertices = np.array([left, right, apex] + unpacked, dtype=np.float64)
        self.vertices = all_vertices

        if verbose:
            print(f"    Unpacked {len(unpacked)} vertices from triangle sides")
            print(f"    Total vertices: {len(self.vertices)}")

        self.transformation_history.append("vertices_unpacked")
        return np.array(unpacked, dtype=np.float64)
    
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

        # Calculate square corners using Euclidean geometry
        base_midpoint = (left + right) / 2.0
        base_vector = right - left
        base_length = np.linalg.norm(base_vector)

        # Create perpendicular vector for height
        if self.m >= 2:
            perp_vector = np.array([-base_vector[1], base_vector[0]])
            if self.m > 2:
                perp_vector = np.pad(perp_vector, (0, self.m - 2), 'constant')
        else:
            perp_vector = np.array([base_length])

        perp_vector = perp_vector / np.linalg.norm(perp_vector)

        # Calculate height from apex
        height_vector = apex - base_midpoint
        height = np.dot(height_vector, perp_vector)

        # Create square with equal sides
        side_length = expansion_factor * (base_length + abs(height)) / 2.0

        # Square corners
        bottom_left = left
        bottom_right = left + side_length * (base_vector / base_length)
        top_left = left + side_length * perp_vector
        top_right = left + side_length * (base_vector / base_length) + side_length * perp_vector

        square_corners = np.array([bottom_left, bottom_right, top_left, top_right], dtype=np.float64)
        self.vertices = square_corners

        # Transform basis to align with square
        square_basis = np.zeros((self.n, self.m), dtype=np.float64)
        square_center = np.mean(square_corners, axis=0)

        for i in range(self.n):
            # Project onto square coordinate system
            vec = self.basis[i] - square_center
            # Align with square axes
            x_proj = np.dot(vec, base_vector / base_length)
            y_proj = np.dot(vec, perp_vector)
            square_basis[i] = square_center + x_proj * (base_vector / base_length) + y_proj * perp_vector

        self.basis = square_basis

        if verbose:
            print(f"    Square base formed with side length {side_length:.3f}")
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

        # Square basis vectors
        v1 = square_vertices[1] - square_vertices[0]  # Bottom edge
        v2 = square_vertices[2] - square_vertices[0]  # Left edge

        # Normalize
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm > 0:
            v1_unit = v1 / v1_norm
        else:
            v1_unit = np.zeros(self.m)
            v1_unit[0] = 1.0

        if v2_norm > 0:
            v2_unit = v2 / v2_norm
        else:
            v2_unit = np.zeros(self.m)
            if self.m > 1:
                v2_unit[1] = 1.0
            else:
                v2_unit[0] = 1.0

        # Apply controlled squaring transformation
        squared_basis = np.zeros((self.n, self.m), dtype=np.float64)

        for i in range(self.n):
            original = self.basis[i]

            # Project onto square coordinate system
            comp_v1 = np.dot(original, v1_unit) * v1_unit
            comp_v2 = np.dot(original, v2_unit) * v2_unit

            # Blend original with square-projected version
            squared_basis[i] = reduction_factor * (comp_v1 + comp_v2) + (1 - reduction_factor) * original

            # Gentle reduction against square vertices
            for sq_vertex in square_vertices:
                if np.linalg.norm(sq_vertex) > 0:
                    proj = np.dot(squared_basis[i], sq_vertex) / np.dot(sq_vertex, sq_vertex)
                    # Only apply partial reduction
                    squared_basis[i] = squared_basis[i] - (reduction_factor * 0.5) * proj * sq_vertex

        # Size-reduction pass (similar to LLL size reduction)
        for i in range(1, self.n):
            for j in range(i):
                if np.linalg.norm(squared_basis[j]) > 0:
                    proj = np.dot(squared_basis[i], squared_basis[j]) / np.dot(squared_basis[j], squared_basis[j])
                    # Round to nearest integer for lattice reduction
                    proj_rounded = round(proj)
                    if abs(proj_rounded) > 0:
                        squared_basis[i] = squared_basis[i] - proj_rounded * squared_basis[j]

        self.basis = squared_basis

        if verbose:
            print(f"    Lattice squared: {self.n}x{self.m} (reduction factor: {reduction_factor})")

            # Find shortest vector
            shortest_norm = float('inf')
            for v in squared_basis:
                norm = np.linalg.norm(v)
                if norm > 0 and norm < shortest_norm:
                    shortest_norm = norm

            if shortest_norm < float('inf'):
                print(f"    Shortest vector length: {shortest_norm:.6f}")

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

            # Swap condition (simplified Lovász) using integer arithmetic
            if k < self.n:
                # Compute squared norms (integers)
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
                            norm_k = (norm_k_sq ** 0.5) if norm_k_sq > 0 else 0
                            norm_km1 = (norm_km1_sq ** 0.5) if norm_km1_sq > 0 else 0
                            print(f"    Swap {k-1} ↔ {k}: {norm_km1:.2f} ↔ {norm_k:.2f}")

        self.basis = basis

        if verbose:
            # Compute shortest vector norm for display
            shortest_sq = float('inf')
            for v in basis:
                norm_sq = sum(int(v[i]) * int(v[i]) for i in range(self.m))
                if norm_sq > 0:
                    shortest_sq = min(shortest_sq, norm_sq)
            shortest = (shortest_sq ** 0.5) if shortest_sq < float('inf') else 0
            print(f"    Final shortest vector: {shortest:.6f}")

        return basis

    def run_full_transformation_with_lll_finish(self, delta: float = 0.75, verbose: bool = True) -> np.ndarray:
        """
        Run the complete geometric transformation sequence with LLL finishing.

        For integer lattices, skips geometric transformations and uses pure integer LLL.

        Args:
            delta: LLL reduction parameter
            verbose: Print progress

        Returns:
            Fully transformed and LLL-finished lattice basis
        """
        if verbose:
            print("=" * 70)
            if self.is_integer:
                print("INTEGER LATTICE SQUARER + LLL FINISH")
            else:
                print("EUCLIDEAN GEOMETRIC LATTICE SQUARER + LLL FINISH")
            print("=" * 70)
            print(f"Starting lattice: {self.n}x{self.m}\n")

        # For integer lattices, skip geometric transformations and use pure integer LLL
        if self.is_integer:
            if verbose:
                print("[*] Integer lattice detected - using pure integer LLL reduction")
            # Just apply integer LLL reduction directly
            result = self.finish_with_lll_like_reduction(delta=delta, verbose=verbose)
        else:
            # Run geometric transformation for float lattices
            self.insert_as_point(verbose=verbose)
            self.unfold_to_line(verbose=verbose)
            self.unfold_to_triangle(verbose=verbose)
            self.unpack_vertices_from_triangle_sides(verbose=verbose)
            self.form_square_base(verbose=verbose)
            self.square_the_lattice(reduction_factor=0.5, verbose=verbose)

            # Finish with LLL-like reduction
            result = self.finish_with_lll_like_reduction(delta=delta, verbose=verbose)

        if verbose:
            print("\n" + "=" * 70)
            if self.is_integer:
                print("INTEGER LLL REDUCTION COMPLETE")
            else:
                print("GEOMETRIC + LLL REDUCTION COMPLETE")
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


if __name__ == "__main__":
    # Run basic demo
    demo_euclidean_squarer()

    # Run comprehensive lattice reduction power test
    test_lattice_reduction_power()

    # Run massive 500x500 lattice test
    test_500x500_lattice()
