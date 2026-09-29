from typing import List


class BaseSymbol:
    """基类：实现自动比较和哈希计算。"""

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __hash__(self):
        # 按照属性名排序后生成元组，若属性值为列表则转换为元组
        items = []
        for key in sorted(self.__dict__.keys()):
            value = self.__dict__[key]
            if isinstance(value, list):
                value = tuple(value)
            items.append((key, value))
        return hash(tuple(items))


class TSymbol(BaseSymbol):
    """
    终结符号（Terminal Symbol）

    Attributes:
        degree: 超边的节点个数
        is_aromatic: 是否处于芳香环中
        symbol: 原子符号
        num_explicit_Hs: 该节点上明确标明的氢原子个数
        formal_charge: 形式电荷
        chirality: 手性信息
    """

    def __init__(self, degree, is_aromatic, symbol, num_explicit_Hs, formal_charge, chirality):
        self.degree = degree
        self.is_aromatic = is_aromatic
        self.symbol = symbol
        self.num_explicit_Hs = num_explicit_Hs
        self.formal_charge = formal_charge
        self.chirality = chirality
        self.terminal = True

    def __str__(self):
        return (f"degree={self.degree}, is_aromatic={self.is_aromatic}, symbol={self.symbol}, "
                f"num_explicit_Hs={self.num_explicit_Hs}, formal_charge={self.formal_charge}, "
                f"chirality={self.chirality}")


class NTSymbol(BaseSymbol):
    """
    非终结符号（Non-terminal Symbol）

    Attributes:
        degree: 超边的度
        is_aromatic: 如果True，至少有一个相关的键是芳香键
        for_ring: 标记是否用于环
        bond_symbol_list: 键符号列表，按 bond_type 升序排序
    """

    def __init__(self, degree: int, is_aromatic: bool, bond_symbol_list: List, for_ring=False):
        self.degree = degree
        self.is_aromatic = is_aromatic
        self.for_ring = for_ring # NOTE: 这个变量好像没有用到，后面可以删掉
        self.terminal = False
        # 记录了atom连接了哪些边， 使用内置sorted函数，根据bond_type排序
        self.bond_symbol_list = sorted(bond_symbol_list, key=lambda bond: bond.bond_type)

    @property
    def symbol(self):
        return 'R'

    def __str__(self) -> str:
        bond_list_str = [str(bond) for bond in self.bond_symbol_list]
        return (f"degree={self.degree}, is_aromatic={self.is_aromatic}, "
                f"bond_symbol_list={bond_list_str}, for_ring={self.for_ring}")


class BondSymbol(BaseSymbol):
    """
    键符号（Bond Symbol）

    Attributes:
        is_aromatic: 是否涉及芳香键
        bond_type: 键的类型
        stereo: 立体化学信息
    """

    def __init__(self, is_aromatic: bool, bond_type: int, stereo: int):
        self.is_aromatic = is_aromatic
        self.bond_type = bond_type
        self.stereo = stereo

    def __str__(self) -> str:
        return (f"is_aromatic={self.is_aromatic}, bond_type={self.bond_type}, "
                f"stereo={self.stereo}, ")



