struct _test{bool empty;};


float test(float b){
      float x;
      float y;
      x = sqrt(sin(b * 2.0) * b) / 10.0;
      y = 33.0 * x + cos(b);
      return y * 10.0;
  }

float getindex(image2D x, ivec2 idx){
      return imageLoad(x,idx).x;
  }

void setindex!(image2D x, vec4 val, ivec2 idx){
      return imageStore(x,idx,val);
  }

void setindex!(image2D x, float val, ivec2 idx){
      return setindex!(x,Vec(val,val,val,val),idx);
  }

void broadcastkernel(_test f, image2D out, image2D A){
      ivec2 idx;
      idx = ivec2(GlobalInvocationID());
      float _ssavalue_0 = test(getindex(A,idx));
      setindex!(out,_ssavalue_0,idx);
      ;
  }
