ctx = opencl(is_gpu)
# ctx = threaded()
img = RGB{Float32}.(load(homedir()*"/test.jpg"));

a = GPUArray(img);
out = similar(a);
k = GPUArray(Float32.(collect(Kernel.gaussian(3))));
imgc = similar(img)
@btime conv!($a, $out, $k);
@btime
@which imfilter!(imgc, img, (Kernel.gaussian(3)))
Array(out)
