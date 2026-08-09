const HBAR2_2ME = 3.81; // eV · Å²  (ħ²/2mₑ)
const HC = 1239.84;     // eV · nm

const MATERIALS = {
  CdSe: {label:"CdSe — Cadmium Selenide", Eg:1.74, me:0.13, mh:0.45},
  PbS:  {label:"PbS — Lead Sulfide",      Eg:0.41, me:0.085, mh:0.085},
  InAs: {label:"InAs — Indium Arsenide",  Eg:0.354, me:0.023, mh:0.41},
  Si:   {label:"Si — Silicon (reference)",Eg:1.12, me:0.26, mh:0.39},
  InP:  {label:"InP — Indium Phosphide (Cd-free)", Eg:1.35, me:0.077, mh:0.60},
  CsPbBr3: {label:"CsPbBr₃ — Perovskite QD (Cd-free)", Eg:2.3, me:0.20, mh:0.20},
  FAPbI3:  {label:"FAPbI₃ — Perovskite QD (Cd-free)",  Eg:1.48, me:0.20, mh:0.20},
};

const SHAPES = {
  circle:  {label:"Circular", icon:'<circle cx="14" cy="14" r="10"/>'},
  square:  {label:"Square",   icon:'<rect x="5" y="5" width="18" height="18"/>'},
  hexagon: {label:"Hexagonal",icon:'<polygon points="14,3 23,8.5 23,19.5 14,25 5,19.5 5,8.5"/>'},
  triangle:{label:"Triangular",icon:'<polygon points="14,4 25,23 3,23"/>'},
};

let state = { shape:'circle' };


// ---------- geometry ----------
function insideShape(shape, x, y, sizeA){
  // x,y in Å relative to center. sizeA = characteristic size in Å.
  const r = sizeA/2;
  switch(shape){
    case 'circle': return x*x+y*y <= r*r;
    case 'square': return Math.abs(x)<=r && Math.abs(y)<=r;
    case 'hexagon': {
      let inside = true;
      for(let k=0;k<6;k++){
        const th = Math.PI/3*k;
        const apo = r*Math.cos(Math.PI/6);
        if (x*Math.cos(th)+y*Math.sin(th) > apo) { inside=false; break; }
      }
      return inside;
    }
    case 'triangle': {
      let inside = true;
      const apo = r*0.5;
      for(let k=0;k<3;k++){
        const th = Math.PI/2 + Math.PI*2/3*k;
        if (x*Math.cos(th)+y*Math.sin(th) > apo) { inside=false; break; }
      }
      return inside;
    }
  }
  return false;
}

function buildGrid(shape, sizeNm, N, V0){
  const sizeA = sizeNm*10;
  const domainA = sizeA*2.4;
  const dx = domainA/(N-1);
  const V = new Float64Array(N*N);
  const c = domainA/2;
  for(let i=0;i<N;i++){
    for(let j=0;j<N;j++){
      const x = i*dx-c, y=j*dx-c;
      V[i*N+j] = insideShape(shape,x,y,sizeA) ? 0 : V0;
    }
  }
  return {V, dx, N};
}

// ---------- matrix-free Hamiltonian ----------
function applyH(psi, N, dx, V, mr, out){
  const t = HBAR2_2ME/(mr*dx*dx);
  for(let i=0;i<N;i++){
    for(let j=0;j<N;j++){
      const idx = i*N+j;
      const c = psi[idx];
      const left  = i>0   ? psi[idx-N] : 0;
      const right = i<N-1 ? psi[idx+N] : 0;
      const up    = j>0   ? psi[idx-1] : 0;
      const down  = j<N-1 ? psi[idx+1] : 0;
      out[idx] = t*(4*c - left-right-up-down) + V[idx]*c;
    }
  }
  return out;
}

function dot(a,b){ let s=0; for(let i=0;i<a.length;i++) s+=a[i]*b[i]; return s; }
function norm(a){ return Math.sqrt(dot(a,a)); }

// small/medium symmetric eigensolver (cyclic Jacobi) — used on the reduced Lanczos matrix
function jacobiEigen(A, n){
  const M = A.map(r=>r.slice());
  const V = Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>i===j?1:0));
  for(let sweep=0; sweep<60; sweep++){
    let off=0;
    for(let p=0;p<n;p++) for(let q=p+1;q<n;q++) off += M[p][q]*M[p][q];
    if(off < 1e-16) break;
    for(let p=0;p<n;p++){
      for(let q=p+1;q<n;q++){
        if(Math.abs(M[p][q]) < 1e-14) continue;
        const theta = (M[q][q]-M[p][p])/(2*M[p][q]);
        const sign = theta>=0?1:-1;
        const t = sign/(Math.abs(theta)+Math.sqrt(theta*theta+1));
        const c = 1/Math.sqrt(t*t+1), s = t*c;
        const app=M[p][p], aqq=M[q][q], apq=M[p][q];
        M[p][p] = c*c*app - 2*s*c*apq + s*s*aqq;
        M[q][q] = s*s*app + 2*s*c*apq + c*c*aqq;
        M[p][q]=0; M[q][p]=0;
        for(let i=0;i<n;i++){
          if(i!==p && i!==q){
            const aip=M[i][p], aiq=M[i][q];
            M[i][p]=c*aip-s*aiq; M[p][i]=M[i][p];
            M[i][q]=s*aip+c*aiq; M[q][i]=M[i][q];
          }
        }
        for(let i=0;i<n;i++){
          const vip=V[i][p], viq=V[i][q];
          V[i][p]=c*vip-s*viq;
          V[i][q]=s*vip+c*viq;
        }
      }
    }
  }
  const vals = M.map((r,i)=>r[i]);
  const idxOrder = vals.map((v,i)=>i).sort((a,b)=>vals[a]-vals[b]);
  const sortedVals = idxOrder.map(i=>vals[i]);
  const sortedVecs = idxOrder.map(ci => V.map(row=>row[ci]));
  return {values:sortedVals, vectors:sortedVecs}; // vectors[c] is array over n for state c
}

// Lanczos iteration (matrix-free) with full reorthogonalization, followed by
// Rayleigh-Ritz extraction of the lowest k eigenpairs from the small tridiagonal matrix.
// Validated against the analytical circular infinite-well solution (Bessel zeros):
// converges to the correct ground/excited energies within a few % by m≈35-50 steps.
function solveLowest(N, dx, V, mr, k, m){
  const dim = N*N;
  let v = new Float64Array(dim);
  for(let i=0;i<dim;i++) v[i] = Math.random()-0.5;
  let nrm = norm(v); for(let i=0;i<dim;i++) v[i] /= nrm;

  const basis = [v.slice()];
  const alphas = [], betas = [];
  let vprev = new Float64Array(dim), betaPrev = 0;
  const tmp = new Float64Array(dim);

  for(let j=0; j<m; j++){
    const vj = basis[j];
    applyH(vj, N, dx, V, mr, tmp);
    let w = tmp.slice();
    if(j>0) for(let i=0;i<dim;i++) w[i] -= betaPrev*vprev[i];
    const alpha = dot(w,vj);
    for(let i=0;i<dim;i++) w[i] -= alpha*vj[i];
    // full reorthogonalization against all previous Lanczos vectors (stability)
    for(let p=0;p<=j;p++){ const d=dot(w,basis[p]); for(let i=0;i<dim;i++) w[i]-=d*basis[p][i]; }
    const beta = norm(w);
    alphas.push(alpha);
    if(beta < 1e-10 || j===m-1){ betas.push(beta); break; }
    betas.push(beta);
    for(let i=0;i<dim;i++) w[i] /= beta;
    basis.push(w);
    vprev = vj; betaPrev = beta;
  }

  const mm = alphas.length;
  const T = Array.from({length:mm}, ()=>new Array(mm).fill(0));
  for(let i=0;i<mm;i++){ T[i][i]=alphas[i]; if(i<mm-1){ T[i][i+1]=betas[i]; T[i+1][i]=betas[i]; } }
  const {values, vectors} = jacobiEigen(T, mm);

  const kk = Math.min(k, mm);
  const energies = values.slice(0, kk);
  const ritz = [];
  for(let c=0;c<kk;c++){
    const psi = new Float64Array(dim);
    for(let p=0;p<mm;p++){ const w = vectors[p][c]; const bp = basis[p]; for(let i=0;i<dim;i++) psi[i]+=w*bp[i]; }
    ritz.push(psi);
  }
  return {energies, vectors: ritz};
}


// ---------- wavelength -> RGB (Bruton approximation) ----------
function wavelengthToRGB(nm){
  let r=0,g=0,b=0;
  if(nm<380) return [80,0,120];
  if(nm>750) return [50,0,0];
  if(nm<440){ r=-(nm-440)/(440-380); g=0; b=1; }
  else if(nm<490){ r=0; g=(nm-440)/(490-440); b=1; }
  else if(nm<510){ r=0; g=1; b=-(nm-510)/(510-490); }
  else if(nm<580){ r=(nm-510)/(580-510); g=1; b=0; }
  else if(nm<645){ r=1; g=-(nm-645)/(645-580); b=0; }
  else { r=1; g=0; b=0; }
  let factor = 1;
  if(nm<420) factor = 0.3+0.7*(nm-380)/(420-380);
  else if(nm>700) factor = 0.3+0.7*(750-nm)/(750-700);
  return [Math.round(255*r*factor), Math.round(255*g*factor), Math.round(255*b*factor)];
}
function regionLabel(nm){
  if(nm<380) return 'Ultraviolet';
  if(nm<450) return 'Violet (visible)';
  if(nm<495) return 'Blue (visible)';
  if(nm<570) return 'Green (visible)';
  if(nm<590) return 'Yellow (visible)';
  if(nm<620) return 'Orange (visible)';
  if(nm<=750) return 'Red (visible)';
  return 'Infrared';
}


module.exports = { HBAR2_2ME, HC, MATERIALS, insideShape, buildGrid, applyH, dot, norm, jacobiEigen, solveLowest, wavelengthToRGB, regionLabel };
