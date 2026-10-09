// Original LightCraft WGSL implementation of the published NAFNet operations.
@group(0) @binding(0) var<storage, read> p: array<u32>;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;
var<workgroup> left: array<f32, 256>;
var<workgroup> right: array<f32, 512>;
var<workgroup> reduction: array<f32, 256>;

// Each lane reuses a weight across two adjacent output pixels. The fixed
// inner reduction is unrolled so shared-memory offsets remain constants.
@compute @workgroup_size(16,16)
fn pointwise(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) g: vec3<u32>) {
    let down=p[4]==2u;
    let n=select(p[1]*p[2],(p[1]/2u)*(p[2]/2u),down);
    let c=select(p[0],p[0]*4u,down); let out=p[3];
    let pixel=g.x*32+l.x*2; let channel=g.y*16+l.y;
    let i=l.y*16+l.x; let r=l.y*32+l.x*2;
    var sum=vec2<f32>(0.0);
    for (var base=0u; base<c; base+=16u) {
        left[i]=0.0; right[r]=0.0; right[r+1]=0.0;
        if channel<out && base+l.x<c { left[i]=weights[channel*c+base+l.x]; }
        if base+l.y<c {
            if pixel<n { right[r]=matrix_input(base+l.y,pixel,n); }
            if pixel+1<n { right[r+1]=matrix_input(base+l.y,pixel+1,n); }
        }
        workgroupBarrier();
        sum += left[l.y*16+0u]*vec2<f32>(right[0u+l.x*2],right[1u+l.x*2]);
        sum += left[l.y*16+1u]*vec2<f32>(right[32u+l.x*2],right[33u+l.x*2]);
        sum += left[l.y*16+2u]*vec2<f32>(right[64u+l.x*2],right[65u+l.x*2]);
        sum += left[l.y*16+3u]*vec2<f32>(right[96u+l.x*2],right[97u+l.x*2]);
        sum += left[l.y*16+4u]*vec2<f32>(right[128u+l.x*2],right[129u+l.x*2]);
        sum += left[l.y*16+5u]*vec2<f32>(right[160u+l.x*2],right[161u+l.x*2]);
        sum += left[l.y*16+6u]*vec2<f32>(right[192u+l.x*2],right[193u+l.x*2]);
        sum += left[l.y*16+7u]*vec2<f32>(right[224u+l.x*2],right[225u+l.x*2]);
        sum += left[l.y*16+8u]*vec2<f32>(right[256u+l.x*2],right[257u+l.x*2]);
        sum += left[l.y*16+9u]*vec2<f32>(right[288u+l.x*2],right[289u+l.x*2]);
        sum += left[l.y*16+10u]*vec2<f32>(right[320u+l.x*2],right[321u+l.x*2]);
        sum += left[l.y*16+11u]*vec2<f32>(right[352u+l.x*2],right[353u+l.x*2]);
        sum += left[l.y*16+12u]*vec2<f32>(right[384u+l.x*2],right[385u+l.x*2]);
        sum += left[l.y*16+13u]*vec2<f32>(right[416u+l.x*2],right[417u+l.x*2]);
        sum += left[l.y*16+14u]*vec2<f32>(right[448u+l.x*2],right[449u+l.x*2]);
        sum += left[l.y*16+15u]*vec2<f32>(right[480u+l.x*2],right[481u+l.x*2]);
        workgroupBarrier();
    }
    if channel<out {
        if p[9]!=0u { sum+=vec2<f32>(weights[p[8]+channel]); }
        if pixel<n { dst[channel*n+pixel]=sum.x; }
        if pixel+1<n { dst[channel*n+pixel+1]=sum.y; }
    }
}

// Stride-two 2x2 downsampling is an implicit matrix product. Gather input
// coordinates directly, avoiding an im2col allocation or another dispatch.
fn matrix_input(ch:u32, pixel:u32, n:u32) -> f32 {
    if p[4]==2u {
        let ow=p[2]/2u; let y=pixel/ow*2u+(ch%4u)/2u; let x=pixel%ow*2u+ch%2u;
        return a[(ch/4u)*p[1]*p[2]+y*p[2]+x];
    }
    return a[ch*n+pixel];
}

@compute @workgroup_size(64)
fn spatial(@builtin(global_invocation_id) id: vec3<u32>) {
    let c=p[0]; let h=p[1]; let w=p[2]; let out=p[3]; let k=p[4]; let stride=p[5]; let pad=p[6];
    let oh=(h+2u*pad-k)/stride+1u; let ow=(w+2u*pad-k)/stride+1u;
    let i=id.x+id.y*2097152u; if i>=out*oh*ow { return; }
    let channel=i/(oh*ow); let y=(i/ow)%oh; let x=i%ow;
    var sum=weights[p[8]+channel];
    for (var ci=0u; ci<c; ci++) { for (var ky=0u; ky<k; ky++) { for (var kx=0u; kx<k; kx++) {
        let sy=i32(y*stride+ky)-i32(pad); let sx=i32(x*stride+kx)-i32(pad);
        if sy>=0 && sx>=0 && sy<i32(h) && sx<i32(w) {
            sum += a[ci*h*w+u32(sy)*w+u32(sx)]*weights[((channel*c+ci)*k+ky)*k+kx];
        }
    } } }
    dst[i]=sum;
}

@compute @workgroup_size(64)
fn depthwise(@builtin(global_invocation_id) id: vec3<u32>) {
    let h=p[1]; let w=p[2]; let i=id.x+id.y*2097152u; if i>=p[0]*h*w { return; }
    let channel=i/(h*w); let y=(i/w)%h; let x=i%w;
    var sum=0.0;
    for (var ky=0u; ky<3u; ky++) { for (var kx=0u; kx<3u; kx++) {
        let sy=i32(y+ky)-1; let sx=i32(x+kx)-1;
        if sy>=0 && sx>=0 && sy<i32(h) && sx<i32(w) {
            sum += a[channel*h*w+u32(sy)*w+u32(sx)]*weights[channel*9u+ky*3u+kx];
        }
    } }
    dst[i]=sum+weights[p[8]+channel];
}

@compute @workgroup_size(64)
fn norm(@builtin(global_invocation_id) id: vec3<u32>) {
    let n=p[1]*p[2]; let pixel=id.x+id.y*2097152u; if pixel>=n { return; }
    let c=p[0]; var mean=0.0; var variance=0.0;
    for (var ch=0u; ch<c; ch++) { mean+=a[ch*n+pixel]; }
    mean/=f32(c);
    for (var ch=0u; ch<c; ch++) { let d=a[ch*n+pixel]-mean; variance+=d*d; }
    let scale=sqrt(variance/f32(c)+1e-6);
    for (var ch=0u; ch<c; ch++) {
        dst[ch*n+pixel]=(a[ch*n+pixel]-mean)/scale*weights[ch]+weights[c+ch];
    }
}

// Deep layers have few pixels and many channels. Each workgroup reduces
// 32 channel lanes for eight adjacent pixels, keeping source loads contiguous.
@compute @workgroup_size(8,32)
fn norm_channels(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) g: vec3<u32>) {
    let n=p[1]*p[2]; let pixel=g.x*8+l.x; let c=p[0];
    let lane=l.y*8+l.x;
    var sum=0.0;
    if pixel<n { for (var ch=l.y; ch<c; ch+=32u) { sum+=a[ch*n+pixel]; } }
    reduction[lane]=sum; workgroupBarrier();
    for (var stride=16u; stride>0u; stride/=2u) {
        if l.y<stride { reduction[lane]+=reduction[lane+stride*8]; }
        workgroupBarrier();
    }
    let mean=reduction[l.x]/f32(c);
    // All lanes read the mean before shared storage is reused for variance.
    workgroupBarrier();
    var variance=0.0;
    if pixel<n { for (var ch=l.y; ch<c; ch+=32u) { let d=a[ch*n+pixel]-mean; variance+=d*d; } }
    reduction[lane]=variance; workgroupBarrier();
    for (var stride=16u; stride>0u; stride/=2u) {
        if l.y<stride { reduction[lane]+=reduction[lane+stride*8]; }
        workgroupBarrier();
    }
    let scale=sqrt(reduction[l.x]/f32(c)+1e-6);
    if pixel<n { for (var ch=l.y; ch<c; ch+=32u) {
        dst[ch*n+pixel]=(a[ch*n+pixel]-mean)/scale*weights[ch]+weights[c+ch];
    } }
}

// AdaptiveAvgPool2d(1), over the whole tile, never a local/sliding pooling approximation.
@compute @workgroup_size(256)
fn pool(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) g: vec3<u32>) {
    let n=p[1]*p[2]; var sum=0.0;
    for (var i=lane; i<n; i+=256u) { sum+=a[g.x*n+i]; }
    reduction[lane]=sum; workgroupBarrier();
    for (var stride=128u; stride>0u; stride/=2u) {
        if lane<stride { reduction[lane]+=reduction[lane+stride]; }
        workgroupBarrier();
    }
    if lane==0u { dst[g.x]=reduction[0]/f32(n); }
}

@compute @workgroup_size(64)
fn map(@builtin(global_invocation_id) id: vec3<u32>) {
    let n=p[1]*p[2]; let i=id.x+id.y*2097152u; let c=p[3]; if i>=c*n { return; }
    let ch=i/n; let mode=p[7];
    if mode==0u { dst[i]=a[i]*a[i+c*n]; } // SimpleGate
    if mode==1u { dst[i]=a[i]*b[ch]; } // simplified channel attention
    if mode==2u { dst[i]=a[i]+b[i]*weights[ch]; } // learned residual scale
    if mode==3u { dst[i]=a[i]+b[i]; }
    if mode==4u { // PixelShuffle(2), p.h/w are the output dimensions
        let y=(i/p[2])%p[1]; let x=i%p[2];
        dst[i]=a[(ch*4u+(y%2u)*2u+x%2u)*(n/4u)+(y/2u)*(p[2]/2u)+x/2u];
    }
}
