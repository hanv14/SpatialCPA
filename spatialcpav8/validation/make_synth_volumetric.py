"""Near-identical volumetric z-planes: same tissue, tiny drift + jitter with z."""
import numpy as np, anndata as ad, pandas as pd
def make(seed=1, n_sections=7, n_per=380, n_genes=60, n_types=5):
    rng=np.random.default_rng(seed)
    type_means=rng.gamma(2.0,1.0,size=(n_types,n_genes)).astype(np.float32)
    # base cell layout shared across z; each z jitters positions slightly & few cells swap type
    r0=np.sqrt(rng.uniform(0,1,n_per))*40; th0=rng.uniform(0,2*np.pi,n_per)
    xs,ys,zs,secs,cts,X=[],[],[],[],[],[]
    for si in range(n_sections):
        z=float(si)
        jit=rng.normal(0,0.8,size=(n_per,2))  # small in-plane jitter (volumetric)
        x=r0*np.cos(th0)+0.3*z+jit[:,0]; y=r0*np.sin(th0)+jit[:,1]
        edges=np.linspace(0,40,n_types+1)+0.3*np.sin(0.2*z)
        t=np.clip(np.digitize(r0,edges)-1,0,n_types-1)
        grad=1.0+0.5*np.cos(0.1*x+0.05*z)
        expr=rng.poisson(np.clip(type_means[t]*grad[:,None],0,None)).astype(np.float32)
        xs.append(x);ys.append(y);zs.append(np.full(n_per,z))
        secs.append(np.array([f"S{si}"]*n_per));cts.append(t);X.append(expr)
    X=np.vstack(X)
    obs=pd.DataFrame({"section":np.concatenate(secs),
                      "cell_type":np.array([f"type_{i}" for i in np.concatenate(cts)])})
    sp=np.column_stack([np.concatenate(xs),np.concatenate(ys),np.concatenate(zs)])
    A=ad.AnnData(X=X,obs=obs,var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]))
    A.obsm["spatial"]=sp.astype(np.float64); A.uns["expression_type"]="raw_counts"
    return A
if __name__=="__main__":
    import sys; A=make(); A.write_h5ad(sys.argv[1]); print("wrote",A.shape)
