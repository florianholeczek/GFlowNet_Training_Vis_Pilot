from dashboard import run_dashboard
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdFMCS
import base64



def imagefn_seh(smiles):
    if smiles is None:
        return None

    def smiles_to_mol(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return mol
        except Exception:
            pass
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is not None:
                return mol
        except Exception:
            pass
        try:
            mol = Chem.MolFromSmarts(smiles)
            if mol is not None:
                return mol
        except Exception:
            pass

        if mol is None:
           return None
    mol = smiles_to_mol(smiles)
    svg = Draw.MolsToGridImage(
        [mol],
        molsPerRow=1,
        subImgSize=(200, 200),
        useSVG=True
    )
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")

    return b64

#state aggregation seh
def state_agg_fn_seh(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    #find mcs
    mcs_result = rdFMCS.FindMCS(mols)
    #convert
    mcs_smarts = mcs_result.smartsString
    mcs_mol = Chem.MolFromSmarts(mcs_smarts)
    mcs_smiles = Chem.MolToSmiles(mcs_mol)
    return mcs_smiles

#plotting function for debugdata
def imagefn_debugdata(s):
    dots = [(i%3, i//3) for i in range(int(s))]
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="60" height="100">' + \
          ''.join(f'<circle cx="{10+x*20}" cy="{10+y*20}" r="5" fill="black"/>' for x,y in dots) + \
          '</svg>'
    b64 = base64.b64encode(svg.encode()).decode()
    return b64

#aggregation function for debugdata
def state_agg_fn_debugdata(texts):
    return min([int(i) for i in texts])



run_dashboard(
    data="seh_small",
    text_to_img_fn=imagefn_seh,
    state_aggregation_fn=state_agg_fn_seh,
    s0="C1=CC=CC=C1",
    debug_mode=True)
#run_dashboard(data="debugdata", text_to_img_fn=imagefn_debugdata, state_aggregation_fn= state_agg_fn_debugdata, debug_mode=True)