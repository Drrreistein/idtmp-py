

from IPython import embed

class CacheNode(object):
    def __init__(self, action=None, motion=None, depth=0):
        self.action = action
        self.motion = motion
        self.depth = depth
        self.children = set()
    
    def __hash__(self) -> int:
        return hash((self.action, self.motion, self.depth))
    
    def __repr__(self):
        return "{}: {}".format(self.__class__, self.__hash__())

class PlanCache(object):
    def __init__(self):
        self.root = CacheNode()

    def add_feasible_motion(self, task_plans:list, motion_plans:list):
        node = self.root
        depth = 0
        for action, motion in zip(task_plans, motion_plans):
            depth += 1
            for child in node.children:
                if action == child.action:
                    node = child
                    break
            else:
                new_node = CacheNode(action, motion, depth)
                node.children.add(new_node)
                node = new_node

    def find_plan_prefixes(self, task_plans:list):
        """
        find pre-validated action in task plan
        """
        depth = -1
        motion_plans = []
        node = self.root
        for action in task_plans:
            for child in node.children:
                if action == child.action:
                    node = child
                    motion_plans.append(child.motion)
                    depth += 1
                    # print(f"found action")
                    break
            else:
                # print(f"find plan prefixed terminate")
                break
        return depth, motion_plans

    def count_tree_node(self, node, count):
        if len(node.children) == 0:
            return 0
        for ch in node.children:
            count += 1
            count = self.count_tree_node(ch, count)
        return count

    def print_node(self, node):
        for ch in node.children:
            s = ' ' * ch.depth * 4
            print(f"{s}{ch.depth}: {ch.action}")
            self.print_node(ch)
    # def recover_motion_plan(self, task_plans:list):
    #     motion_plans = []
    #     return motion_plans
