
from IPython import embed

def enumerate_tree(root):
    """
    enumerate etamp tree, to check if a sampled-place was repeatedly branched
    """
    if len(root.children) == 0:
        return None
    for ch in root.children:
        if ch.parent is not None and 'sample-place' in ch.parent.pddl:
            indent_str = ' ' * ch.depth * 2
            if ch.add_mapping is not None:
                print(f"{indent_str}{ch.depth}: {list(ch.add_mapping.values())[0].value.pose[0][:2]}")
        enumerate_tree(ch)

def enumerate_tree2(root):
    """
    enumerate etamp tree, to check if a sampled-place was repeatedly branched, breath-first search
    """

    if len(root.children) == 0:
        return None
    for i,ch in enumerate(root.children):
        if ch.parent is not None and 'sample-place' in ch.parent.pddl:
            indent_str = ' ' * ch.depth * 2
            if ch.add_mapping is not None:
                print(f"{indent_str}{ch.depth}: {list(ch.add_mapping.values())[0].value.pose[0][:2]}")
        if i==len(root.children)-1:
            print('')
    for ch in root.children:
        enumerate_tree2(ch)
