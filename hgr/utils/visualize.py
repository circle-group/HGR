from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
import io
from PIL import Image

def visualize_mol(mol, size=(300, 300), add_idx=True, show=True):
    """
    可视化一个 RDKit 分子（仅内存操作，不写文件）。

    参数:
        mol: RDKit Mol 对象或 SMILES 字符串
        size: 图像大小 (width, height)
        add_idx: 是否在原子上显示原子索引
        show: 是否直接展示 (matplotlib)

    返回:
        对于SVG返回字符串，对于PNG返回字节流
    """


    # 1. 如果是字符串，先转为Mol对象
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)

    # 2. 给原子设置AtomMapNum作为编号
    if add_idx:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())

    # 3. 绘制（这里用 Cairo 模式画成PNG）
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    tm = rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer.DrawMolecule(tm)
    drawer.FinishDrawing()

    img_bytes = drawer.GetDrawingText()

    if show:
        # 4. 直接用matplotlib展示
        img = Image.open(io.BytesIO(img_bytes))
        plt.figure(figsize=(size[0] / 100, size[1] / 100))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    return img_bytes  # 返回绘制的字节流（png格式）


def visualize_mol_svg(mol, size=(300,300), add_idx=True, show=True, save_svg=None):
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
        if mol is None:
            raise ValueError("Invalid SMILES")

    if add_idx:
        for a in mol.GetAtoms():
            a.SetAtomMapNum(a.GetIdx())

    d = rdMolDraw2D.MolDraw2DSVG(*size)
    tm = rdMolDraw2D.PrepareMolForDrawing(mol)
    d.DrawMolecule(tm)
    d.FinishDrawing()
    svg = d.GetDrawingText()

    if save_svg:
        with open(save_svg, "w", encoding="utf-8") as f:
            f.write(svg)
        print(f"SVG saved to: {save_svg}")


    # 在 PyCharm/Jupyter 里可直接预览 SVG 文件；或用浏览器打开
    return svg

if __name__ == '__main__':
    # Example usage
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    # visualize_mol(mol, size=(400, 400), add_idx=True, show=True)
    # 上面一行代码在集群上可以，下面这在mac上可以运行
    visualize_mol_svg(mol, size=(400, 400), add_idx=True, save_svg='../analysis/figs/test.svg')

    # Test the function with a SMILES string

