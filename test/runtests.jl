# test/runtests.jl

using OtsuThresholding # Load the module defined in src/
using Test
using Images, ColorTypes, FixedPointNumbers # For creating test images
using Statistics, LinearAlgebra # Might be needed by underlying code or helpers implicitly
using StatsBase, Optim # Dependencies of the function being tested

println("--- Running Otsu Thresholding Tests ---")

@testset "OtsuThresholding.jl" begin

    @testset "Basic Grayscale Cases" begin
        # Simple 2-level image
        img1 = Float64[0.1 0.1 0.2; 0.8 0.9 0.9]
        IDX1, sep1 = otsu(img1) # n=2 default
        @test size(IDX1) == size(img1)
        @test eltype(IDX1) == Int
        # Expect separation between low (0.1, 0.2) and high (0.8, 0.9) values
        @test Set(unique(IDX1)) == Set([1, 2])
        @test IDX1[1, 1:3] == [1, 1, 1] # Low values should be class 1
        @test IDX1[2, 1:3] == [2, 2, 2] # High values should be class 2
        @test 0.9 <= sep1 <= 1.0 # Should be near perfect separation

        # Simple 3-level image (ensure distinct levels)
        img2 = Float64[0.1 0.1; 0.5 0.5; 0.9 0.9]
        IDX2, sep2 = otsu(img2, 3)
        @test size(IDX2) == size(img2)
        @test Set(unique(IDX2)) == Set([1, 2, 3])
        @test IDX2[1, :] == [1, 1]
        @test IDX2[2, :] == [2, 2]
        @test IDX2[3, :] == [3, 3]
        @test 0.9 <= sep2 <= 1.0 # Near perfect separation

        # Simple 4-level image (tests optimization path)
        img3 = Float64[0.1 0.1; 0.3 0.3; 0.6 0.6; 0.9 0.9]
        IDX3, sep3 = otsu(img3, 4)
        @test size(IDX3) == size(img3)
        @test Set(unique(IDX3)) == Set([1, 2, 3, 4])
        @test IDX3[1, :] == [1, 1]
        @test IDX3[2, :] == [2, 2]
        @test IDX3[3, :] == [3, 3]
        @test IDX3[4, :] == [4, 4]
        @test 0.9 <= sep3 <= 1.0 # Near perfect separation

        # Test with UInt8 input
        img4 = UInt8[10 10; 200 200]
        IDX4, sep4 = otsu(img4)
        @test size(IDX4) == size(img4)
        @test Set(unique(IDX4)) == Set([1, 2])
        @test IDX4 == [1 1; 2 2]
        @test 0.9 <= sep4 <= 1.0
    end

    @testset "Input Type Handling" begin
        # Gray type input
        img_gray = Gray{N0f8}[0.1 0.1; 0.9 0.9] # N0f8 is FixedPointNumber for 0-1 range
        IDX_g, sep_g = otsu(img_gray)
        @test size(IDX_g) == size(img_gray)
        @test Set(unique(IDX_g)) == Set([1, 2])
        @test IDX_g == [1 1; 2 2]
        @test 0.9 <= sep_g <= 1.0

        # 2D RGB type input
        img_rgb2d = RGB{N0f8}[RGB(0.1,0,0) RGB(0,0.1,0); RGB(0,0,0.9) RGB(0.9,0.9,0)]
        # KLT makes exact IDX hard to predict, check valid output
        IDX_rgb2d, sep_rgb2d = otsu(img_rgb2d, 3)
        @test size(IDX_rgb2d) == size(img_rgb2d)
        @test eltype(IDX_rgb2d) == Int
        @test 0.0 <= sep_rgb2d <= 1.0
        @test all(x -> 0 <= x <= 3, IDX_rgb2d) # Classes 0, 1, 2, 3 possible

        # 3D Numeric RGB input (Float64)
        img_rgb3d_f = zeros(2, 2, 3)
        img_rgb3d_f[1, 1, 1] = 0.8 # Red channel high
        img_rgb3d_f[2, 2, 3] = 0.9 # Blue channel high
        IDX_rgb3d_f, sep_rgb3d_f = otsu(img_rgb3d_f, 2)
        @test size(IDX_rgb3d_f) == (2, 2)
        @test eltype(IDX_rgb3d_f) == Int
        @test 0.0 <= sep_rgb3d_f <= 1.0
        @test all(x -> 0 <= x <= 2, IDX_rgb3d_f)

        # 3D Numeric RGB input (UInt8)
        img_rgb3d_u = zeros(UInt8, 2, 2, 3)
        img_rgb3d_u[1, :, 1] .= 200 # Row 1 predominantly red
        img_rgb3d_u[2, :, 3] .= 220 # Row 2 predominantly blue
        IDX_rgb3d_u, sep_rgb3d_u = otsu(img_rgb3d_u, 2)
        @test size(IDX_rgb3d_u) == (2, 2)
        @test eltype(IDX_rgb3d_u) == Int
        @test 0.0 <= sep_rgb3d_u <= 1.0
        @test all(x -> 0 <= x <= 2, IDX_rgb3d_u)
        # Expect row separation after KLT
        @test IDX_rgb3d_u[1,1] == IDX_rgb3d_u[1,2]
        @test IDX_rgb3d_u[2,1] == IDX_rgb3d_u[2,2]
        @test IDX_rgb3d_u[1,1] != IDX_rgb3d_u[2,1]
    end

    @testset "Edge Cases" begin
        # n = 1
        img_n1 = [0.1 0.9; 0.2 0.8]
        IDX_n1, sep_n1 = otsu(img_n1, 1)
        @test size(IDX_n1) == size(img_n1)
        @test IDX_n1 == ones(Int, 2, 2)
        @test sep_n1 == 0.0

        # Constant image
        img_const = fill(0.5, 3, 3)
        IDX_c2, sep_c2 = otsu(img_const, 2) # n=2
        @test size(IDX_c2) == size(img_const)
        @test IDX_c2 == ones(Int, 3, 3) # All assigned to class 1
        @test sep_c2 == 0.0

        IDX_c3, sep_c3 = otsu(img_const, 3) # n=3
        @test size(IDX_c3) == size(img_const)
        @test IDX_c3 == ones(Int, 3, 3) # All assigned to class 1
        @test sep_c3 == 0.0

        # Image with fewer levels than n
        img_few = [0.1 0.1; 0.9 0.9]
        # Use @test_warn macro to check for the specific warning message
        warn_msg_few = r"Image has only 2 distinct intensity levels.*fewer than.*n=3"
        IDX_few, sep_few = @test_warn warn_msg_few otsu(img_few, 3)
        @test size(IDX_few) == size(img_few)
        # Should assign classes based on available levels (1 and 2)
        @test Set(unique(IDX_few)) == Set([1, 2])
        @test IDX_few == [1 1; 2 2]
        @test sep_few == 0.0 # Separability is 0 when nbins < n

        # Image with exactly n levels
        img_exact = [0.1 0.1; 0.5 0.5; 0.9 0.9]
        warn_msg_exact = r"Image has exactly n=3 distinct intensity levels"
        IDX_exact, sep_exact = @test_warn warn_msg_exact otsu(img_exact, 3)
        @test size(IDX_exact) == size(img_exact)
        @test Set(unique(IDX_exact)) == Set([1, 2, 3])
        @test IDX_exact == [1 1; 2 2; 3 3]
        @test sep_exact == 1.0 # Separability should be 1

        # Image with NaN/Inf
        img_naninf = Float64[0.1 NaN Inf; 0.9 0.8 0.9]
        IDX_ni, sep_ni = otsu(img_naninf, 2)
        @test size(IDX_ni) == size(img_naninf)
        @test Set(unique(IDX_ni)) == Set([0, 1, 2]) # Includes 0 for non-finite
        @test IDX_ni[1, 1] == 1 # 0.1 -> class 1
        @test IDX_ni[1, 2] == 0 # NaN -> class 0
        @test IDX_ni[1, 3] == 0 # Inf -> class 0
        @test IDX_ni[2, 1] == 2 # 0.9 -> class 2
        @test IDX_ni[2, 2] == 2 # 0.8 -> class 2 (likely)
        @test IDX_ni[2, 3] == 2 # 0.9 -> class 2
        @test 0.0 <= sep_ni <= 1.0

        # Empty finite mask case (e.g., all NaN image)
        img_all_nan = fill(NaN, 2, 2)
        IDX_an, sep_an = otsu(img_all_nan, 2)
        @test size(IDX_an) == size(img_all_nan)
        @test all(IDX_an .== 0)
        @test sep_an == 0.0
    end

    @testset "Argument Validation" begin
        img_dummy = [0.1 0.9; 0.1 0.9]
        # n <= 0
        @test_throws ErrorException otsu(img_dummy, 0)
        @test_throws ErrorException otsu(img_dummy, -1)

        # n > 256 (should warn and cap at 256)
        warn_msg_high_n = r"n \(257\) is high.*Capping n at 256"
        IDX_hn, sep_hn = @test_warn warn_msg_high_n otsu(img_dummy, 257)
        @test size(IDX_hn) == size(img_dummy) # Should still process
        @test 0.0 <= sep_hn <= 1.0

        # Unsupported input dimension/type
        img_4d = ones(2, 2, 3, 4) # 4D array
        @test_throws ErrorException otsu(img_4d, 2)
        img_3d_gray = ones(2, 2, 1) # 3D grayscale
         @test_throws ErrorException otsu(img_3d_gray, 2)
        img_complex = [1.0+im 2.0; 3.0 4.0] # Complex numbers
        # Error message might vary depending on where it fails (isfinite, type check, etc.)
        @test_throws MethodError otsu(img_complex, 2) # Likely method error earlier now
    end
end
