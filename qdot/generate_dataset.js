const fs = require('fs');
const P = require('./physics_core.js');

const shapes = ['circle','square','hexagon','triangle'];
const materials = Object.keys(P.MATERIALS);

const N = 32;      // moderate grid for data-gen speed (surrogate learns the trend, exact solver stays ground truth for the real Solve button)
const LANCZOS_M = 36;

function sample(min, max){ return min + Math.random()*(max-min); }

const rows = [];
const header = 'shape,size_nm,V0_eV,Eg_bulk,me,mh,Ee0,Eh0\n';
let count = 0;
const t0 = Date.now();

for(const shape of shapes){
  for(const matKey of materials){
    const mat = P.MATERIALS[matKey];
    const nSamples = 130; // per shape x material (7 materials now — keep total dataset size reasonable)
    for(let s=0; s<nSamples; s++){
      const size = sample(2, 14);      // nm
      const V0 = sample(0.3, 4.0);     // eV

      const {V, dx} = P.buildGrid(shape, size, N, V0);
      const e = P.solveLowest(N, dx, V, mat.me, 1, LANCZOS_M);
      const h = P.solveLowest(N, dx, V, mat.mh, 1, LANCZOS_M);
      const Ee0 = e.energies[0], Eh0 = h.energies[0];

      rows.push(`${shape},${size.toFixed(4)},${V0.toFixed(4)},${mat.Eg},${mat.me},${mat.mh},${Ee0.toFixed(6)},${Eh0.toFixed(6)}`);
      count++;
    }
    console.log(`done ${shape}/${matKey}  (${count} total, ${((Date.now()-t0)/1000).toFixed(1)}s)`);
  }
}

fs.writeFileSync('dataset.csv', header + rows.join('\n') + '\n');
console.log('Wrote', count, 'rows to dataset.csv in', ((Date.now()-t0)/1000).toFixed(1), 's');
