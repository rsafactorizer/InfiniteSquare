# Geometric Lattice Factorization Tool

A novel approach to integer factorization using geometric transformations on 3D lattice structures. This tool explores the mathematical relationship between spatial compression and number theory.

## Overview

The Lattice Tool implements a unique factorization strategy that encodes integers into 3D lattice coordinates and applies a series of geometric transformations to compress the lattice from a cube down to a single point. Through recursive refinement over multiple iterations, the method attempts to extract factors from the compressed coordinate space.

### Key Concept

The approach is based on the idea that:
1. A composite number N = p × q can be encoded as a point in 3D lattice space
2. Geometric compression preserves certain structural relationships
3. Multiple iterations of micro-lattice creation and collapse refine the search space
4. Factor extraction uses GCD operations on the recursively refined coordinates

## Features

- **3D Lattice Representation**: Full cube-based lattice structure (e.g., 100×100×100)
- **Geometric Transformations**: Multi-stage compression sequence
  - Volume → Plane → Line → Square → Triangle → Line → Point
- **Recursive Refinement**: Iterative zoom with ~100 refinement cycles
- **Precision Preservation**: No scaling loss during encoding
- **High-Dimensional Remainder Encoding**: 3D remainder lattice (100×100×100) for enhanced resolution
- **Multiple Extraction Methods**: GCD-based, coordinate scaling, and modular arithmetic approaches

## Usage

### Basic Factorization

```python
from Squarer import factor_with_lattice_compression

# Factor a number
result = factor_with_lattice_compression(15)
# Output: factors found, compression metrics, final compressed point
```

### Command Line

```bash
# Factor a specific number
python Squarer.py 15

# Run demo with custom lattice size
python Squarer.py 100

# Default demo (tries multiple test numbers)
python Squarer.py
```

### Advanced Usage

```python
# Specify custom lattice size
result = factor_with_lattice_compression(323, lattice_size=200)

# Access detailed results
factors = result['factors']  # List of (p, q) tuples
metrics = result['compression_metrics']  # Compression statistics
final_point = result['final_point']  # Final compressed coordinates
```

## How It Works

### Stage 0: Encoding
- Encodes N as lattice coordinates: x = a mod L, y = b mod L
- Uses 3D remainder lattice for precision preservation
- No scaling applied to maintain GCD relationships

### Stage A: Macro-Collapse
- Creates initial lattice cube (e.g., 100×100×100)
- Applies transformation sequence:
  1. Compress 3D volume to 2D plane
  2. Expand point to line
  3. Create square (+ shape)
  4. Form bounded square
  5. Add vertex lines (diagonals)
  6. Compress to triangle
  7. Compress to line
  8. Compress to point (singularity)

### Stages B & C: Recursive Refinement (×100 iterations)
- **Stage B**: Create new micro-lattice (100×100×100) centered on previous compressed point
- **Stage C**: Collapse the micro-lattice through full transformation sequence
- Each iteration refines search space by factor of 10^6
- After 100 iterations: cumulative refinement of 10^600

### Factor Extraction
Multiple methods attempt to extract factors from the compressed coordinates:
1. **Coordinate scaling with zoom factor**
2. **3D remainder lattice GCD extraction**
3. **Modular arithmetic relationships**
4. **Sum/difference patterns**
5. **Direct search around refined coordinates**

## Algorithm Complexity

- **Space**: O(L³) for lattice size L
- **Time per iteration**: O(L³) transformations
- **Total iterations**: ~100 refinement cycles
- **Search window**: Dramatically reduced from O(√N) to highly refined space

## Examples

```python
# Small composite numbers
factor_with_lattice_compression(15)   # 3 × 5
factor_with_lattice_compression(21)   # 3 × 7
factor_with_lattice_compression(35)   # 5 × 7

# Larger semiprimes
factor_with_lattice_compression(143)  # 11 × 13
factor_with_lattice_compression(323)  # 17 × 19
factor_with_lattice_compression(2021) # 43 × 47
```

## Limitations

- **Experimental approach**: Not proven to be polynomial-time
- **Success rate**: Variable depending on number structure
- **Large numbers**: May require extensive search ranges
- **Memory**: Scales with lattice size (L³ points)

## Implementation Details

### Classes

- **`LatticePoint`**: Represents a point in integer lattice coordinates (x, y, z)
- **`LatticeLine`**: Represents a line segment between two lattice points
- **`GeometricLattice`**: Main class implementing the transformation sequence

### Key Methods

- `compress_volume_to_plane()`: 3D → 2D compression
- `expand_point_to_line()`: Point → Line expansion
- `create_square_from_line()`: Line → Square transformation
- `compress_square_to_triangle()`: Square → Triangle compression
- `compress_triangle_to_line()`: Triangle → Line compression
- `compress_line_to_point()`: Final compression to singularity

## Mathematical Background

The method explores connections between:
- Geometric compression and information preservation
- Modular arithmetic and GCD relationships
- Lattice point density and factor structure
- Recursive refinement and search space reduction

## Contributing

Contributions welcome! Areas for exploration:
- Alternative encoding schemes
- Different transformation sequences
- Optimization of search strategies
- Theoretical analysis of convergence properties

## License

MIT License - see LICENSE file for details

## Disclaimer

This is an experimental mathematical exploration. The method is not proven to provide polynomial-time factorization and should not be relied upon for cryptographic applications.

**Note**: This tool represents a novel geometric approach to factorization. While it demonstrates interesting mathematical properties through lattice compression and recursive refinement, its effectiveness for large-scale factorization (e.g., RSA-2048) remains an open research question.
