__global__ void rotate_kernel(unsigned char *input, unsigned char *output, int width, int height, int batch_size, float angle) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int img_idx = blockIdx.z;  // Each image has a separate z-dimension block

    if (img_idx < batch_size && x < width && y < height) {
        // Calculate the center of the image
        int cx = width / 2;
        int cy = height / 2;

        // Calculate the new coordinates after rotation
        float newX = cos(angle) * (x - cx) - sin(angle) * (y - cy) + cx;
        float newY = sin(angle) * (x - cx) + cos(angle) * (y - cy) + cy;

        // Check bounds and apply rotation for this batch image
        if (newX >= 0 && newX < width && newY >= 0 && newY < height) {
            int idx = (img_idx * height * width + y * width + x) * 3; // original index
            int newIdx = (img_idx * height * width + int(newY) * width + int(newX)) * 3; // new index
            
            // Rotate the RGB channels
            output[newIdx] = input[idx];       // Red channel
            output[newIdx + 1] = input[idx + 1]; // Green channel
            output[newIdx + 2] = input[idx + 2]; // Blue channel
        }
    }
}