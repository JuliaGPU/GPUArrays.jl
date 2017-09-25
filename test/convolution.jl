using GPUArrays, Colors, FileIO, ImageFiltering
using CLArrays
using GPUArrays: synchronize_threads
import GPUArrays: LocalMemory
using CLArrays


img = RGB{Float32}.(load(homedir()*"/Desktop/backround.jpg"));

a = CLArray(img);
out = similar(a);
k = CLArray(Float32.(collect(Kernel.gaussian(3))));
imgc = similar(img)

# convolution!(a, out, k);
# Array(out)
# outc = similar(img)
# copy!(outc, out)
