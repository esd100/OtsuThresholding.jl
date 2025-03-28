#=
OtsuThresholding.jl - Multi-level Otsu Thresholding in Julia

Copyright (c) 2007-2017 Damien Garcia
Original MATLAB implementation. Source link (if known): (https://www.biomecardio.com/matlab/otsu.m)

This Julia version is a translation and adaptation.
Maintained by: <Eric Diaz> <eric.diaz@gmail.com>
Copyright (c) <2025> <Eric Diaz>

Translation/refactoring assistance provided by AI (Google Gemini 2.5 Pro Experimental 03-25).

Licensed under the MIT License (see LICENSE file).
=#

module OtsuThresholding

# --- Required Packages ---
# Ensure these are added to your project environment:
using StatsBase
using LinearAlgebra
using Statistics
using Optim
# Note: Images, TestImages, ImageView etc. are needed for running examples,
# but not strictly for the otsu function itself if you load images differently.
using Images 
using ColorTypes # Often needed by Images

# Makes main function available when someone uses `using OtsuThresholding`
export otsu

# --- Helper Functions ---

"""
    _isrgb(A::AbstractArray) -> Bool

Internal helper to check if an array represents an RGB image.
Checks for:
 - 3D numeric array (UInt8/UInt16/Float) with 3 channels in 3rd dim.
 - 2D array of RGB structs (from ColorTypes.jl).
Validates value range [0, 1] for floats.
"""
function _isrgb(A::AbstractArray)
    T = eltype(A)
    D = ndims(A)

    # Case 1: 2D array of RGB structs
    if T <: RGB && D == 2
        # Assume valid if type matches ColorTypes.jl RGB struct
        return true
    end

    # Case 2: 3D numeric array
    # Check dimensions and type
    if D == 3 && size(A, 3) == 3 && T <: Union{UInt8, UInt16, AbstractFloat}
        # Check float range [0, 1] if applicable
        if T <: AbstractFloat
            finite_vals = A[isfinite.(A)]
            if isempty(finite_vals)
                # Treat fully non-finite as not valid range
                return false
            end
            min_val, max_val = extrema(finite_vals)
            # Check if approximately within [0, 1] allowing for small tolerance
            if (min_val < 0.0 && !isapprox(min_val, 0.0, atol=1e-6)) || 
               (max_val > 1.0 && !isapprox(max_val, 1.0, atol=1e-6))
                return false # Outside [0, 1] range
            end
            # Float is in range [0, 1]
            return true 
        else # UInt8, UInt16 are implicitly valid range
            return true
        end
    end
    
    # Otherwise, not considered a supported RGB format by this function
    return false
end

"""
    _klt_grayscale(I_rgb::AbstractArray{<:Real, 3}) -> Matrix{Float32}
    _klt_grayscale(I_rgb::AbstractArray{<:RGB, 2}) -> Matrix{Float32}

Internal helper to convert an RGB image to grayscale using Karhunen-Loève Transform (PCA),
keeping the component with the highest energy (variance). Handles both standard 3D arrays
and 2D arrays of RGB structs.
Returns a Float32 matrix.
"""
function _klt_grayscale(I_rgb::AbstractArray{T, D}) where {T, D} # Generic for AbstractArray
    
    local I_vec::Matrix{Float32}
    local orig_size::Tuple

    if T <: RGB && D == 2 # Handle 2D array of RGB type
        orig_size = size(I_rgb)
         # Convert RGB structs to a 3-channel Float32 array first
         # Manually extract components
         R = Float32.(red.(I_rgb))
         G = Float32.(green.(I_rgb))
         B = Float32.(blue.(I_rgb))
         # Reshape each channel to a column vector and concatenate
         I_vec_r = reshape(R, :)
         I_vec_g = reshape(G, :)
         I_vec_b = reshape(B, :)
         I_vec = hcat(I_vec_r, I_vec_g, I_vec_b) # N_pixels x 3 matrix

    elseif !(T <: RGB) && D == 3 && size(I_rgb, 3) == 3 # Handle standard 3D array
        # Promote element type to at least Float32 for calculations
        I_rgb_float = T <: AbstractFloat ? Float32.(I_rgb) : Float32.(I_rgb) ./ (T == UInt8 ? 255.0f0 : T == UInt16 ? 65535.0f0 : 1.0f0)
        orig_size = (size(I_rgb_float, 1), size(I_rgb_float, 2))
        # Reshape to N_pixels x 3
        I_vec = reshape(I_rgb_float, :, 3)
    else
         error("_klt_grayscale expects a 2D array of RGB types or a 3D array with size(A,3)==3.")
    end

    # Center the data (optional but standard for PCA)
    mean_vec = mean(I_vec, dims=1)
    I_vec_centered = I_vec .- mean_vec
    
    # Calculate covariance matrix (ensure correct dimensions for cov)
    # cov expects variables in columns, observations in rows
    C = cov(I_vec_centered) # Use centered data for covariance
    
    # Eigen decomposition
    # Use Symmetric wrapper for potentially better numerical stability/speed
    eigen_decomp = eigen(Symmetric(C)) 
    eigenvalues = eigen_decomp.values
    eigenvectors = eigen_decomp.vectors
    
    # Find component with max eigenvalue (max energy)
    # Eigenvalues should be real and non-negative for Symmetric(cov)
    max_eigval, max_idx = findmax(real.(eigenvalues)) # Use real part just in case
    
    # Project centered data onto the principal component
    principal_component = I_vec_centered * eigenvectors[:, max_idx]
    
    # Reshape back to 2D image
    I_gray = reshape(principal_component, orig_size[1], orig_size[2])
    
    # Output is already Float32 due to Float32 promotion/calculations
    return I_gray
end


# --- Main Otsu Function ---

"""
    otsu(I::AbstractArray{<:Union{Real, Colorant}}, n::Integer=2) -> (Matrix{Int}, Float64)
    otsu(I::AbstractArray{<:Union{Real, Colorant}}) -> (Matrix{Int}, Float64)

Global image thresholding/segmentation using Otsu's N-level method.

Segments the image `I` into `n` classes by maximizing the between-class variance.

# Arguments
- `I`: Input image. Can be a 2D grayscale array (subtype of `Real` or `Gray`), 
       or a 3D RGB array (e.g., `Array{Float64, 3}`) or 2D array of `RGB` types. 
       Input types like `UInt8`, `UInt16`, or `AbstractFloat` are expected.
       RGB images are converted to grayscale using PCA (KLT) first.
- `n`: The desired number of classes (thresholds + 1). Default is 2.

# Returns
- `IDX::Matrix{Int}`: An array of the same size as the input (or the grayscale version),
                     containing cluster indices (from 1 to `n`) for each pixel.
                     `0` is assigned to non-finite pixels (NaN or Inf) in the input `I`.
- `sep::Float64`: The value of the separability criterion (between-class variance / total variance),
                  within the range [0, 1]. 0 indicates issues (e.g., fewer unique values than `n`),
                  1 indicates perfect separability (only achieved with exactly `n` distinct values).

# Details
- The function first handles RGB input by converting it to grayscale using the principal component
  with the most energy (variance) via KLT/PCA. Grayscale images (`Gray` type) are converted to standard numeric arrays.
- The grayscale image intensity values are then normalized and mapped to 0-255 integer levels.
- Otsu's method is applied to the histogram of these levels.
- For `n=2` and `n=3`, direct formulas are used.
- For `n >= 4`, optimization (`Optim.optimize` with Nelder-Mead) is used to find the `n-1`
  thresholds that maximize the separability criterion.
- It's noted that thresholds generally become less credible as `n` increases.

# References
- Otsu N, "A Threshold Selection Method from Gray-Level Histograms,"
  IEEE Trans. Syst. Man Cybern., vol. SMC-9, no. 1, pp. 62-66, 1979.
  DOI: 10.1109/TSMC.1979.4310076

# Example
```julia
using Images, TestImages, ImageView # Ensure ImageView is installed
img = testimage("mandril_gray") # Example grayscale image

# 2-level thresholding (default)
IDX2, sep2 = otsu(img)
println("Separability for n=2: ", sep2)
imshow(IDX2, name="Otsu n=2") # Requires ImageView

# 3-level thresholding
IDX3, sep3 = otsu(img, 3)
println("Separability for n=3: ", sep3)
imshow(IDX3, name="Otsu n=3")

# 4-level thresholding
IDX4, sep4 = otsu(img, 4)
println("Separability for n=4: ", sep4)
imshow(IDX4, name="Otsu n=4")

# Example with RGB image (e.g., from TestImages)
img_rgb = testimage("mandrill")
IDX_rgb, sep_rgb = otsu(img_rgb, 3)
println("Separability for RGB (n=3): ", sep_rgb)
imshow(IDX_rgb, name="Otsu RGB n=3")

"""
function otsu(I_orig::AbstractArray{<:Union{Real, Colorant}}, n::Integer = 2)
    # --- Argument Validation ---
if n == 1
    # Return index 1 for all finite pixels, 0 for non-finite
    IDX = ones(Int, size(I_orig))
    # Handle potential non-real elements if Colorant input has issues
    try 
        finite_mask = isfinite.(I_orig)
        IDX[.!finite_mask] .= 0
    catch # Broad catch if isfinite fails on type
         # Assume all are class 1 if isfinite check fails
    end
    return IDX, 0.0 # Separability is ill-defined or 0
elseif n <= 0 # n != round(n) is implicitly true for Integer type
    error("n must be a strictly positive integer!")
elseif n > 256 # Keep 256 limit consistent with 8-bit histogram approach
    @warn "n ($n) is high, Otsu might be less effective. Capping n at 256."
    n = 256
end

# --- Input Processing ---
# Store non-finite mask from original input - handle types carefully
# We will use this ORIGINAL mask only at the VERY END.
local original_finite_mask
try
    if eltype(I_orig) <: Union{Real, Gray} 
         numeric_I = eltype(I_orig) <: Gray ? Float32.(Gray.(I_orig)) : I_orig
         original_finite_mask = isfinite.(numeric_I)
    else 
        original_finite_mask = trues(size(I_orig)) # Default assumption
         # If needed, add more sophisticated checks for NaNs in components here
    end
catch e
     @warn "Could not determine original finite mask accurately for input type $(eltype(I_orig)). Assuming all finite. Error: $e"
     original_finite_mask = trues(size(I_orig))
end

local I::Matrix{Float32} # Ensure I is Matrix{Float32} after processing

# Check for RGB and convert using KLT/PCA if necessary
if _isrgb(I_orig)
    I = _klt_grayscale(I_orig)
elseif eltype(I_orig) <: Gray && ndims(I_orig) == 2
     I = Float32.(Gray.(I_orig)) 
elseif eltype(I_orig) <: Real && ndims(I_orig) == 2
    I = Float32.(I_orig)
else
    error("Input must be a 2D grayscale array (Real or Gray type), a 2D RGB array, or a 3D numeric RGB image.")
end

# Keep track of original size for IDX output (size of the 2D array I)
output_size = size(I) # output_size is 2D

# --- RECALCULATE finite_mask based on the processed 2D grayscale image I ---
finite_mask = isfinite.(I) # This is now guaranteed to be 2D

# Handle empty or fully non-finite input (using the 2D mask)
if !any(finite_mask)
    # If no finite values in processed I, return zeros respecting original non-finite if possible
    IDX = zeros(Int, output_size)
    # If original_finite_mask has same 2D size, use it, otherwise just return all zeros
    if size(original_finite_mask) == output_size
        IDX[.!original_finite_mask] .= 0 # Ensure original NaNs are 0 if sizes match
    end
    return IDX, 0.0
end

# --- Normalization and Histogramming ---
# Use the 2D finite_mask from here onwards
finite_vals_gray = I[finite_mask] 
# Need to handle case where finite_vals_gray might be empty if !any(finite_mask)
# This is now checked above.
min_val, max_val = extrema(finite_vals_gray)

# Normalize finite values to 0-255 range
I_norm = zeros(UInt8, output_size) # Initialize output array
delta = max_val - min_val

if delta < eps(Float32) # Handle constant image case
    # All finite pixels get a middle gray value? Or 0? Let's use 0.
    # The histogram will have one peak.
    I_norm[finite_mask] .= 0 
else
    # Apply normalization only to finite values
    scale = 255.0f0 / delta
    # Use broadcasting assignment for efficiency
    I_norm[finite_mask] .= round.(UInt8, (I[finite_mask] .- min_val) .* scale)
end

# Calculate histogram using StatsBase on the 0-255 values
# Use edges 0:256 for bins representing values 0, 1, ..., 255
# Consider only finite values for histogram calculation
histo = fit(Histogram, I_norm[finite_mask], 0:256).weights
nbins = 256 # We are working with 256 bins (values 0-255)
pixel_values = 0:(nbins-1) # Intensity values corresponding to bins

# Check if number of unique intensity levels is sufficient
nbins_actual = count(>(0), histo)

if nbins_actual == 0 # Should not happen if finite_mask had true values
     return zeros(Int, output_size), 0.0
elseif nbins_actual < n
    @warn "Image has only $nbins_actual distinct intensity levels (after normalization), fewer than the requested n=$n classes. Assigning based on existing levels."
    IDX = zeros(Int, output_size)
    present_values = sort(pixel_values[histo .> 0]) # Ensure sorted order
    val_map = Dict(present_values[i] => i for i in 1:nbins_actual)
    # Assign class index based on the normalized value for finite pixels
    vals_to_map = I_norm[finite_mask]
    mapped_indices = [get(val_map, val, 0) for val in vals_to_map]
    IDX[finite_mask] .= mapped_indices
    return IDX, 0.0 # Separability is 0
elseif nbins_actual == n
     @warn "Image has exactly n=$n distinct intensity levels (after normalization). Assigning each level to a class."
     IDX = zeros(Int, output_size)
     present_values = sort(pixel_values[histo .> 0]) # Ensure sorted order
     val_map = Dict(present_values[i] => i for i in 1:n)
     vals_to_map = I_norm[finite_mask]
     mapped_indices = [get(val_map, val, 0) for val in vals_to_map]
     IDX[finite_mask] .= mapped_indices
     return IDX, 1.0 # Perfect separability by definition
end

# --- Otsu Calculation ---
P = Float64.(histo) ./ sum(histo) # Probability distribution (as Float64 for precision)

# Zeroth and first-order cumulative moments
w = cumsum(P)
# Ensure pixel_values is treated as vector
mu = cumsum(collect(pixel_values) .* P) # Use collect for type stability if pixel_values is range

# Total mean and variance (precompute for n>=4 and separability)
muT = mu[end]
# Avoid NaN from 0*log(0) etc; use P directly
sigma2T = sum(((collect(pixel_values) .- muT).^2) .* P) 

# Handle zero total variance (constant image after normalization)
if sigma2T < eps()
    # All finite pixels belong to class 1
    IDX = zeros(Int, output_size)
    IDX[finite_mask] .= 1
    # Separability is maximal (1) if n=1 (handled earlier),
    # or 0 if n > 1, as no separation is possible/needed.
    return IDX, 0.0 
end

IDX = zeros(Int, output_size) # Initialize output IDX
maxsig = 0.0 # Placeholder for max between-class variance
threshold_vals_opt = Int[] # To store optimal thresholds

# --- n=2 Case (Binary Thresholding) ---
if n == 2
    # Calculate between-class variance (sigma^2_B)
    # Indexing: w and mu have length nbins (256). We test thresholds between bins.
    # A threshold k means splitting after bin k (value k). Tested k goes from 0 to nbins-2.
    # Corresponding indices in w, mu are 1 to nbins-1.
    sigma2B_k = zeros(Float64, nbins - 1)
    for k_idx = 1:(nbins-1) # k_idx corresponds to bin index k_idx (value k_idx-1)
        wk = w[k_idx]        # Pr(class 0) for threshold at k_idx
        muk = mu[k_idx]      # Mean times Pr(class 0) for threshold at k_idx
        
        # Avoid division by zero if wk or (1-wk) is zero
        if wk > eps() && (1.0 - wk) > eps()
             # Formula: (muT * w(k) - mu(k))^2 / (w(k) * (1 - w(k)))
             sigma2B_k[k_idx] = (muT * wk - muk)^2 / (wk * (1.0 - wk))
        # else, variance is 0, already initialized
        end
    end

    # Find threshold k that maximizes sigma2B
    maxsig, k_opt_idx = findmax(sigma2B_k)
    
    # Optimal threshold VALUE is pixel_values[k_opt_idx] (which is k_opt_idx-1)
    threshold_val = pixel_values[k_opt_idx] # Threshold is > this value
    threshold_vals_opt = [threshold_val] # Store the single threshold

# --- n=3 Case ---
elseif n == 3
    max_sigma2B = -1.0
    k1_opt = -1
    k2_opt = -1

    # Thresholds k1, k2 split bins:
    # Class 1: 0 to pixel_values[k1_idx]
    # Class 2: pixel_values[k1_idx]+1 to pixel_values[k2_idx]
    # Class 3: pixel_values[k2_idx]+1 to nbins-1 (pixel_values[nbins])
    # Iterate k1_idx from 1 to nbins-2, k2_idx from k1_idx+1 to nbins-1
    for k1_idx in 1:(nbins-2) 
         w0 = w[k1_idx]
         if w0 < eps() continue end 
         mu0_term = mu[k1_idx]
         mu0 = mu0_term / w0

        for k2_idx in (k1_idx + 1):(nbins-1) 
            w1 = w[k2_idx] - w[k1_idx]
            if w1 < eps() continue end 
            mu1_term = mu[k2_idx] - mu[k1_idx]
            mu1 = mu1_term / w1
            
            # w2 = 1.0 - w[k2_idx] # Or sum P from k2_idx+1 to end
            w2 = w[end] - w[k2_idx] # Use w[end] which is 1.0
            if w2 < eps() continue end 
            mu2_term = mu[end] - mu[k2_idx]
            mu2 = mu2_term / w2

            # Between-class variance
            sigma2B = w0 * (mu0 - muT)^2 + w1 * (mu1 - muT)^2 + w2 * (mu2 - muT)^2
            
            if sigma2B > max_sigma2B
                max_sigma2B = sigma2B
                k1_opt = k1_idx
                k2_opt = k2_idx
            end
        end
    end
    
    maxsig = max_sigma2B
    # Threshold values are the pixel values corresponding to the *last* bin in the class
    threshold1_val = pixel_values[k1_opt] # Class 1 includes this value
    threshold2_val = pixel_values[k2_opt] # Class 2 includes this value
    threshold_vals_opt = [threshold1_val, threshold2_val]
    
# --- n >= 4 Case (Optimization) ---
else # n >= 4
    
    # Objective function to MINIMIZE (1 - sigma2B/sigma2T)
    # Input k_norm is vector of n-1 normalized thresholds [0, 1]
    function sig_func(k_norm::Vector{Float64})
        # Convert normalized thresholds to actual pixel value thresholds (0 to nbins-1)
        # Ensure thresholds are sorted and unique after rounding
        # Clamp to ensure they are valid indices 0 to nbins-1
        threshold_indices = sort(unique(clamp.(round.(Int, k_norm .* (nbins - 1)), 0, nbins-1)))

        # Check if we have the correct number of unique thresholds
        if length(threshold_indices) < n - 1
            return 1.0 # Return max penalty if thresholds collapse
        end

        # Add boundaries for class calculation (using indices)
        # Threshold index t means class ends at bin t+1 (value t)
        # Class j goes from index k_full[j]+1+1 to k_full[j+1]+1
        k_full_indices = [-1; threshold_indices; nbins - 1] # Indices: -1 to nbins-1

        sigma2B_local = 0.0
        for j = 1:n
            # Indices in the P vector (1-based)
            idx_start = k_full_indices[j] + 1 + 1 # Bin index (1-based)
            idx_end = k_full_indices[j+1] + 1   # Bin index (1-based)
            
            # Handle empty range possibility if indices are consecutive
            if idx_start > idx_end
                 wj = 0.0
                 muj = 0.0 # Assign default value
            else
                # Slice P and pixel_values using the 1-based indices
                P_slice = P[idx_start:idx_end]
                pixel_values_slice = pixel_values[idx_start:idx_end]
                
                wj = sum(P_slice)

                if wj < eps()
                    # If class is empty, penalty
                     return 1.0 
                end
                # Calculate mean for this class
                muj = sum(pixel_values_slice .* P_slice) / wj
            end

            if wj >= eps() # Only add variance if class is not empty
                sigma2B_local += wj * (muj - muT)^2
            elseif n > 1 # Penalize empty class if n > 1
                 # We already returned 1.0 if wj < eps, this might be redundant
                 # but emphasizes that empty classes for n>1 are bad
                 return 1.0
            end
        end
        
        # Return the value to be minimized
        # sigma2T was checked earlier to be > eps()
        cost = 1.0 - (sigma2B_local / sigma2T)
        # Ensure cost is not NaN/Inf if something went wrong
        return isfinite(cost) ? cost : 1.0 
    end # end of sig_func

    # Initial guess for normalized thresholds
    k0 = Float64.(1:(n-1)) ./ n
    
    # Optimization settings - TolX equivalent from MATLAB
    opt_tol = 1.0 / (nbins -1) # Tolerance on normalized scale
    # Use Fminbox to constrain thresholds to [0, 1] - safer than Nelder-Mead alone
    lower_bounds = zeros(n-1)
    upper_bounds = ones(n-1)
    # Inner optimizer for Fminbox, e.g., NelderMead
    inner_optimizer = NelderMead() 
    opt_options = Optim.Options(x_tol=opt_tol, f_tol=1e-8, iterations=1000) 

    # Run optimizer
    # result = optimize(sig_func, k0, NelderMead(), opt_options) # Original
    result = optimize(sig_func, lower_bounds, upper_bounds, k0, Fminbox(inner_optimizer), opt_options)
    
    if !Optim.converged(result)
        @warn "Optimization for n=$n did not fully converge. Result might be suboptimal. Reason: $(Optim.summary(result))"
    end

    # Get optimal normalized thresholds and value
    k_norm_opt = Optim.minimizer(result)
    min_y = Optim.minimum(result)
    
    # Convert back to actual thresholds (pixel values 0 to nbins-1)
    # Clamp just in case optimizer slightly violates bounds
    threshold_vals_opt_raw = clamp.(round.(Int, k_norm_opt .* (nbins - 1)), 0, nbins-1)
    threshold_vals_opt = sort(unique(threshold_vals_opt_raw))
    
    # Ensure we still have n-1 thresholds after optimization/rounding/unique
    if length(threshold_vals_opt) < n - 1
         @warn "Optimization resulted in $(length(threshold_vals_opt)) distinct thresholds (needed $(n-1)) for n=$n. Segmentation might combine classes."
         # Strategy: Pad with intermediate or endpoint values if needed?
         # For simplicity, we proceed with the thresholds found. The segmentation loop below handles it.
         # If padding is desired:
         # current_thresholds = Set(threshold_vals_opt)
         # full_range = Set(0:(nbins-1))
         # potential_new = sort(collect(setdiff(full_range, current_thresholds)))
         # while length(threshold_vals_opt) < n - 1 && !isempty(potential_new)
         #     # Add potentially reasonable thresholds, e.g., midpoints or just endpoints
         #     # This logic can get complex. Using what we have is simpler.
         #      push!(threshold_vals_opt, pop!(potential_new)) # Example: add highest available
         #      threshold_vals_opt = sort(unique(threshold_vals_opt))
         # end
    end
    # Ensure the result used has exactly n-1 thresholds for the loop below if possible
    if length(threshold_vals_opt) > n-1
        threshold_vals_opt = threshold_vals_opt[1:n-1] # Should not happen with unique
    end
    # Recalculate max sigma based on the final thresholds for consistency
    final_cost = sig_func(k_norm_opt) # Or threshold_vals_opt / (nbins-1)
    maxsig = (1.0 - final_cost) * sigma2T

end # End of if/elseif n

# --- Final Segmentation Step (using threshold_vals_opt) ---
# This part applies the found thresholds (for n=2, 3, or >=4)
num_thresholds = length(threshold_vals_opt)

if num_thresholds > 0
    # Assign class 1
    IDX[finite_mask .& (I_norm .<= threshold_vals_opt[1])] .= 1
    # Assign intermediate classes
    for i = 1:(num_thresholds-1)
        IDX[finite_mask .& (I_norm .> threshold_vals_opt[i]) .& (I_norm .<= threshold_vals_opt[i+1])] .= i + 1
    end
    # Assign last class (class n = num_thresholds + 1)
    IDX[finite_mask .& (I_norm .> threshold_vals_opt[num_thresholds])] .= num_thresholds + 1
elseif n > 1 # Case where no thresholds were found but n > 1 (e.g., constant image handled earlier, but maybe edge case)
     IDX[finite_mask] .= 1 # Assign all to class 1
# else n=1 was handled at start
end

# --- Finalization ---
# Calculate final separability criterion
sep = (sigma2T > eps()) ? (maxsig / sigma2T) : 0.0
# Clamp sep to [0, 1] just in case of numerical issues
sep = clamp(sep, 0.0, 1.0)

# --- Ensure non-finite pixels in ORIGINAL input are 0 in output IDX ---
# Apply the ORIGINAL finite mask at the very end.
# Need to handle if original_finite_mask was 3D and output_size is 2D
if size(original_finite_mask) == output_size
    # If original was 2D (grayscale), apply its mask directly
    IDX[.!original_finite_mask] .= 0
elseif ndims(original_finite_mask) == 3 && size(original_finite_mask)[1:2] == output_size
    # If original was 3D RGB, check if ANY channel was non-finite for a pixel
    # Create a 2D mask: true if all channels were finite, false otherwise
    # Note: all(original_finite_mask, dims=3) returns a 2x2x1 array, need to reshape/drop dim
    original_2d_finite_mask = reshape(all(original_finite_mask, dims=3), output_size)
    IDX[.!original_2d_finite_mask] .= 0
end
# Pixels that became non-finite during KLT/processing are already handled by `finite_mask` used internally

return IDX, sep

end # end of otsu function

"""
--- Example Usage Section (Optional - keep commented out or remove for library use) ---
#=

Ensure ImageView is loaded if you uncomment this section
using Images, TestImages, ImageView
println("--- Otsu Example Script ---")

Example 1: Grayscale Image
println("\nProcessing grayscale image (mandril_gray)...")
img_gray_test = testimage("mandril_gray")

println("Running Otsu with n=2...")
IDX2_test, sep2_test = otsu(img_gray_test)
println("Separability (n=2): ", sep2_test)
println("Output size: ", size(IDX2_test))
println("Unique classes found: ", sort(unique(IDX2_test)))
try; imshow(IDX2_test, name="Otsu n=2"); catch; println("ImageView unavailable?"); end

println("\nRunning Otsu with n=3...")
IDX3_test, sep3_test = otsu(img_gray_test, 3)
println("Separability (n=3): ", sep3_test)
println("Unique classes found: ", sort(unique(IDX3_test)))
try; imshow(IDX3_test, name="Otsu n=3"); catch; end

println("\nRunning Otsu with n=4...")
IDX4_test, sep4_test = otsu(img_gray_test, 4)
println("Separability (n=4): ", sep4_test)
println("Unique classes found: ", sort(unique(IDX4_test)))
try; imshow(IDX4_test, name="Otsu n=4"); catch; end

Example 2: RGB Image
println("\nProcessing RGB image (mandrill)...")
img_rgb_test = testimage("mandrill")

println("Running Otsu with n=3...")
IDX_rgb_test, sep_rgb_test = otsu(img_rgb_test, 3)
println("Separability (RGB, n=3): ", sep_rgb_test)
println("Output size: ", size(IDX_rgb_test))
println("Unique classes found: ", sort(unique(IDX_rgb_test)))
try; imshow(IDX_rgb_test, name="Otsu RGB n=3"); catch; end

println("\n--- End of Examples ---")
=#
"""
end
