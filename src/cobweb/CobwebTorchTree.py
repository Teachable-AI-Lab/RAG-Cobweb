##-------------------------------------------------------------
## CobwebTorchTree.py
## Implementation of a CobwebTree from https://github.com/Teachable-AI-Lab/cobweb
##-------------------------------------------------------------

import json
import math
from random import shuffle, random
from math import log, isclose
from collections import defaultdict
import heapq

import torch
from src.utils.constants import COBWEB_GREEDY_MODE
from src.cobweb.CobwebTorchNode import CobwebTorchNode

class CobwebTorchTree(object):
    """
    The CobwebTree contains the knowledge base of a particular instance of the
    cobweb algorithm and can be used to fit and categorize instances.
    """

    def __init__(self, shape, use_info=True, acuity_cutoff=False,
                 use_kl=True, prior_var=None, alpha=1e-8, device=None):
        """
        The tree constructor.
        """
        self.device = device
        self.use_info = use_info
        self.acuity_cutoff = acuity_cutoff
        self.use_kl = use_kl
        self.shape = shape
        self.alpha = torch.tensor(alpha, dtype=torch.float, device=self.device,
                                  requires_grad=False)
        self.pi_tensor = torch.tensor(math.pi, dtype=torch.float,
                                      device=self.device, requires_grad=False)

        self.prior_var = prior_var
        if prior_var is None:
            self.prior_var = 1 / (2 * math.e * self.pi_tensor)

        self.clear()

    def clear(self):
        """
        Clears the concepts of the tree and resets the node map.
        """
        self.root = CobwebTorchNode(shape=self.shape, device=self.device)
        self.root.tree = self
        self.labels = {}
        self.reverse_labels = {}
        # Build node_map for id->node lookup
        self.node_map = {self.root.id: self.root}

    def _build_node_map(self):
        """
        Rebuilds the node_map by traversing the tree.
        """
        self.node_map = {}
        def recurse(node):
            self.node_map[node.id] = node
            for c in node.children:
                recurse(c)
        recurse(self.root)

    def __str__(self):
        return str(self.root)

    def dump_json(self):
        # only save reverse labels, because regular labels get converted into
        # strings regardless of type and we know the type of the indices.
        tree_params = {
                'use_info': self.use_info,
                'acuity_cutoff': self.acuity_cutoff,
                'use_kl': self.use_kl,
                'shape': self.shape.tolist() if isinstance(self.shape, torch.Tensor) else self.shape,
                'alpha': self.alpha.item(),
                'prior_var': self.prior_var.item(),
                'reverse_labels': self.reverse_labels}

        json_output = json.dumps(tree_params)[:-1]
        json_output += ', "root": '
        json_output += self.root.iterative_output_json()
        json_output += "}"

        return json_output

    def load_json_helper(self, node_data_json):
        node = CobwebTorchNode(self.shape, device=self.device)
        node.count = torch.tensor(node_data_json['count'], dtype=torch.float,
                                  device=self.device, requires_grad=False)
        node.mean = torch.tensor(node_data_json['mean'], dtype=torch.float,
                                 device=self.device, requires_grad=False)
        node.meanSq = torch.tensor(node_data_json['meanSq'], dtype=torch.float,
                                   device=self.device, requires_grad=False)
        node.label_counts = torch.tensor(node_data_json['label_counts'],
                                         dtype=torch.float, device=self.device,
                                         requires_grad=False)
        node.total_label_count = node.label_counts.sum()
        return node

    def load_json(self, json_string):
        data = json.loads(json_string)

        self.use_info = data['use_info']
        self.acuity_cutoff = data['acuity_cutoff']
        self.use_kl = data['use_kl']
        self.shape = data['shape']
        self.alpha = torch.tensor(data['alpha'], dtype=torch.float,
                                  device=self.device, requires_grad=False)
        self.prior_var = torch.tensor(data['prior_var'], dtype=torch.float,
                                      device=self.device, requires_grad=False)
        self.reverse_labels = {int(attr): data['reverse_labels'][attr] for attr in data['reverse_labels']}
        self.labels = {self.reverse_labels[attr]: attr for attr in self.reverse_labels}
        self.root = self.load_json_helper(data['root'])
        self.root.tree = self

        queue = [(self.root, c) for c in data['root']['children']]

        while len(queue) > 0:
            parent, curr_data = queue.pop()
            curr = self.load_json_helper(curr_data)
            curr.tree = self
            curr.parent = parent
            parent.children.append(curr)

            for c in curr_data['children']:
                queue.append((curr, c))

    def ifit(self, instance, label=None):
        """
        Incrementally fit a new instance into the tree and return its resulting
        concept.

        The instance is passed down the cobweb tree and updates each node to
        incorporate the instance. **This process modifies the tree's
        knowledge** for a non-modifying version of labeling use the
        :meth:`CobwebTree.categorize` function.

        :param instance: An instance to be categorized into the tree.
        :type instance:  :ref:`Instance<instance-rep>`
        :return: A concept describing the instance
        :rtype: CobwebNode

        .. seealso:: :meth:`CobwebTree.cobweb`
        """
        if label is not None and label not in self.labels:
            idx = len(self.labels)
            self.labels[label] = idx
            self.reverse_labels[idx] = label

        with torch.no_grad():
            return self.cobweb(instance, label)

    def fit(self, instances, labels=None, iterations=1, randomize_first=True):
        """
        Fit a collection of instances into the tree.

        This is a batch version of the ifit function that takes a collection of
        instances and categorizes all of them. The instances can be
        incorporated multiple times to burn in the tree with prior knowledge.
        Each iteration of fitting uses a randomized order but the first pass
        can be done in the original order of the list if desired, this is
        useful for initializing the tree with specific prior experience.

        :param instances: a collection of instances
        :type instances:  [:ref:`Instance<instance-rep>`,
            :ref:`Instance<instance-rep>`, ...]
        :param iterations: number of times the list of instances should be fit.
        :type iterations: int
        :param randomize_first: whether or not the first iteration of fitting
            should be done in a random order or in the list's original order.
        :type randomize_first: bool
        """
        if labels is None:
            labels = [None for i in range(len(instances))]

        instance_labels = [(inst, labels[i]) for i, inst in
                           enumerate(instances)]

        for x in range(iterations):
            if x == 0 and randomize_first:
                shuffle(instances)
            for inst, label in instances:
                self.ifit(inst, label)
            shuffle(instances)

    def cobweb(self, instance, label):
        """
        The core cobweb algorithm used in fitting and categorization.

        In the general case, the cobweb algorithm entertains a number of
        sorting operations for the instance and then commits to the operation
        that maximizes the :meth:`category utility
        <CobwebNode.category_utility>` of the tree at the current node and then
        recurses.

        At each node the alogrithm first calculates the category utility of
        inserting the instance at each of the node's children, keeping the best
        two (see: :meth:`CobwebNode.two_best_children
        <CobwebNode.two_best_children>`), and then calculates the
        category_utility of performing other operations using the best two
        children (see: :meth:`CobwebNode.get_best_operation
        <CobwebNode.get_best_operation>`), commiting to whichever operation
        results in the highest category utility. In the case of ties an
        operation is chosen at random.

        In the base case, i.e. a leaf node, the algorithm checks to see if
        the current leaf is an exact match to the current node. If it is, then
        the instance is inserted and the leaf is returned. Otherwise, a new
        leaf is created.

        .. note:: This function is equivalent to calling
            :meth:`CobwebTree.ifit` but its better to call ifit because it is
            the polymorphic method siganture between the different cobweb
            family algorithms.

        :param instance: an instance to incorporate into the tree
        :type instance: :ref:`Instance<instance-rep>`
        :return: a concept describing the instance
        :rtype: CobwebNode

        .. seealso:: :meth:`CobwebTree.ifit`, :meth:`CobwebTree.categorize`
        """
        current = self.root

        while current:
            # the current.count == 0 here is for the initially empty tree.
            if not current.children and (current.is_exact_match(instance, label) or
                                         current.count == 0):
                # print("leaf match")
                current.increment_counts(instance, label)
                break

            elif not current.children:
                # print("fringe split")
                new = CobwebTorchNode(shape=self.shape, device=self.device, otherNode=current)
                current.parent = new
                new.children.append(current)

                if new.parent:
                    new.parent.children.remove(current)
                    new.parent.children.append(new)
                else:
                    self.root = new

                new.increment_counts(instance, label)
                current = new.create_new_child(instance, label)
                break

            else:
                best1_pu, best1, best2 = current.two_best_children(instance,
                                                                   label)

                if not COBWEB_GREEDY_MODE:
                    _, best_action = current.get_best_operation(instance, label, best1,
                                                                best2, best1_pu)
                else:
                    best_action = "best"

                # print(best_action)
                if best_action == 'best':
                    current.increment_counts(instance, label)
                    current = best1
                elif best_action == 'new':
                    current.increment_counts(instance, label)
                    current = current.create_new_child(instance, label)
                    break
                elif best_action == 'merge':
                    current.increment_counts(instance, label)
                    new_child = current.merge(best1, best2)
                    current = new_child
                elif best_action == 'split':
                    current.split(best1)
                else:
                    raise Exception('Best action choice "' + best_action +
                                    '" not a recognized option. This should be'
                                    ' impossible...')

        return current

    def cobweb_categorize_fast(self, instance, label=None, retrieve_k=1):
        """
        A fast version of categorize for cobweb that optimizes the search for
        the sentence node, optimizing and taking the max of P(C|X).

        logP(C|X) = softmax(logP(X|C)) --> softmax is monotonically increasing

        Notes:
        - variance = meanSq / count for each node

        TODO:
        - need to figure out a way to make this procedural! this can be done in
          the long-term but it's almost as easy as creating and iterating on two
          matrices, smartly replacing nodes along the list
          - this is a relatively easy fix EXCEPT for the fact that

        For this quick implementation, we compute the matrix at every call (can
        be very slow over time) and apply the math formulas to identify the best
        node.
        - Potential fixes include only traversing leaf nodes (because those
          are the only relevant nodes to search, leaves are where the sentences
          populate) --> I've implemented this version of the function but can
          extend to all nodes if applicable

        """

        all_nodes = []
        all_leaves = []

        def dfs_tree(node):
            # administrative behavior
            all_nodes.append(node)
            if len(node.children) == 0:
                all_leaves.append(node)
            for child in node.children:
                dfs_tree(child)

        dfs_tree(self.root)

        C_mean = torch.zeros(len(all_leaves), self.shape[0]) # matrix of means, (N, D)
        C_var = torch.zeros(len(all_leaves), self.shape[0]) # matrix of variances = matrix of meanSq / counts (N, D)

        for i, node in enumerate(all_leaves):
            C_mean[i] += node.mean
            C_var[i] += node.meanSq / node.count
            if torch.zeros(1024).equal(node.meanSq):
                if hasattr(node, "sentence_id"):
                    print("LEAF NODE")
                    print(node.sentence_id)
                else:
                    print("NOT LEAF NODE")

        x = instance # the input embeddings vector

        x = x.unsqueeze(0)  # (1, D)
        log_probs = -0.5 * (
            torch.log(2 * torch.pi * C_var).sum(dim=1) +
            ((x - C_mean) ** 2 / C_var).sum(dim=1)
        )

        print(log_probs)

        # find max of log_probs and then find corresponding index of node
        _, topk_indices = torch.topk(log_probs, retrieve_k)

        return [all_leaves[idx] for idx in topk_indices]

    def _cobweb_categorize(self, instance, label, use_best, greedy, max_nodes, retrieve_k=None):
        """
        A cobweb specific version of categorize, not intended to be
        externally called.

        .. seealso:: :meth:`CobwebTree.categorize`
        """
        queue = []
        heapq.heappush(queue, (-self.root.log_prob(instance, label), 0.0, random(), self.root))
        nodes_visited = 0

        best = self.root
        best_score = float('-inf')

        retrieved = []

        while len(queue) > 0:
            neg_score, neg_curr_ll, _, curr = heapq.heappop(queue)
            score = -neg_score # the heap sorts smallest to largest, so we flip the sign
            curr_ll = -neg_curr_ll # the heap sorts smallest to largest, so we flip the sign
            nodes_visited += 1
            curr.update_label_count_size()

            if score > best_score:
                best = curr
                best_score = score

            # CURR_LL is a weird thing for some reason
            # if use_best and best_score > curr_ll:
            #     # if best_score is greater than curr_ll, then we know we've
            #     # found the best and can stop early.
            #     break

            if greedy:
                queue = []

            if nodes_visited >= max_nodes:
                break

            if hasattr(curr, "sentence_id"):
                heapq.heappush(retrieved, (score, random(), curr))

            if retrieve_k is not None and len(retrieved) == retrieve_k:
                break # TODO can replace this with a part at the end optionally!

            if len(curr.children) > 0:
                ll_children_unnorm = torch.zeros(len(curr.children))
                for i, c in enumerate(curr.children):
                    log_prob = c.log_prob(instance, label)
                    ll_children_unnorm[i] = (log_prob + math.log(c.count) - math.log(curr.count))
                log_p_of_x = torch.logsumexp(ll_children_unnorm, dim=0)

                for i, c in enumerate(curr.children):
                    child_ll = ll_children_unnorm[i] - log_p_of_x + curr_ll
                    child_ll_inst = c.log_prob(instance, label)
                    score = child_ll + child_ll_inst # p(c|x) * p(x|c)
                    # score = child_ll # p(c|x)
                    heapq.heappush(queue, (-score, -child_ll, random(), c))

        if retrieve_k is None:
            return best if use_best else curr

        return [retrieved[i][-1] for i in range(retrieve_k)]

    def categorize(self, instance, label=None, use_best=True, greedy=False, max_nodes=float('inf'), retrieve_k=None):
        """
        Sort an instance in the categorization tree and return its resulting
        concept.

        The instance is passed down the categorization tree according to the
        normal cobweb algorithm except using only the best operator and without
        modifying nodes' probability tables. **This process does not modify the
        tree's knowledge** for a modifying version of labeling use the
        :meth:`CobwebTree.ifit` function

        :param instance: an instance to be categorized into the tree.
        :type instance: :ref:`Instance<instance-rep>`
        :return: A concept describing the instance
        :rtype: CobwebNode

        .. seealso:: :meth:`CobwebTree.cobweb`
        """
        with torch.no_grad():
            return self._cobweb_categorize(instance, label, use_best, greedy, max_nodes, retrieve_k)

    def old_categorize(self, instance, label):
        """
        A cobweb specific version of categorize, not intended to be
        externally called.

        .. seealso:: :meth:`CobwebTree.categorize`
        """
        current = self.root

        while True:
            if (len(current.children) == 0):
                return current

            parent = current
            current = None
            best_score = None

            for child in parent.children:
                score = child.log_prob_class_given_instance(instance, label)

                if ((current is None) or ((best_score is None) or (score > best_score))):
                    best_score = score
                    current = child

    def predict_probs(self, instance, label=None, greedy=False,
                      max_nodes=float('inf')):
        with torch.no_grad():
            return self._predict_probs(instance, label, greedy, max_nodes)

    def _predict_probs(self, instance, label, greedy, max_nodes):
        queue = []
        heapq.heappush(queue, (-self.root.log_prob(instance, label), 0.0, random(), self.root))
        nodes_visited = 0

        log_weighted_scores = []
        # total_w = 0

        while len(queue) > 0:
            neg_score, neg_curr_ll, _, curr = heapq.heappop(queue)
            score = -neg_score # the heap sorts smallest to largest, so we flip the sign
            curr_ll = -neg_curr_ll # the heap sorts smallest to largest, so we flip the sign
            nodes_visited += 1

            # w = math.exp(score)
            # total_w += w

            curr.update_label_count_size()
            log_weighted_scores.append(score + torch.log(curr.label_counts) -
                                       torch.log(curr.label_counts.sum()))

            # p_label = {self.reverse_labels[i]: v.item()
            #            for i, v in enumerate(p_label)}

            # for label in p_label:
            #     pred[label].append(w + p_label[label])

            if greedy:
                queue = []

            if nodes_visited >= max_nodes:
                break

            if len(curr.children) > 0:
                ll_children_unnorm = torch.zeros(len(curr.children))
                for i, c in enumerate(curr.children):
                    log_prob = c.log_prob(instance, label)
                    ll_children_unnorm[i] = (log_prob + math.log(c.count) - math.log(curr.count))
                log_p_of_x = torch.logsumexp(ll_children_unnorm, dim=0)

                for i, c in enumerate(curr.children):
                    child_ll = ll_children_unnorm[i] - log_p_of_x + curr_ll
                    child_ll_inst = c.log_prob(instance, label)
                    score = child_ll + child_ll_inst
                    heapq.heappush(queue, (-score, -child_ll, random(), c))

        log_weighted_scores = torch.stack(log_weighted_scores)
        ll = torch.logsumexp(log_weighted_scores, 0) - torch.logsumexp(log_weighted_scores.flatten(), 0)
        pred = {self.reverse_labels[i]: v.item() for i, v in enumerate(torch.exp(ll))}

        # for label in pred:
        #     pred[label] /= total_w

        return pred

    def compute_var(self, meanSq, count):
        # return (meanSq + 30*1) / (count + 30)

        if self.acuity_cutoff:
            return torch.clamp(meanSq / count, self.prior_var) # with cutoff
        else:
            return meanSq / count + self.prior_var # with adjustment

    def compute_score(self, mu1, var1, p_label1, mu2, var2, p_label2):
        if (self.use_info):
            if (self.use_kl):
                # score2 = (0.5 * (torch.log(var2) - torch.log(var1)) +
                #          (var1 + torch.pow(mu1 - mu2, 2))/(2 * var2) -
                #          0.5).sum()
                score = (torch.log(var2) - torch.log(var1)).sum()
                score += ((var1 + torch.pow(mu1 - mu2, 2))/(var2)).sum()
                score -= mu1.numel()
                score /= 2

                # if torch.abs(score - score2) > 1e-3:
                #     print(score - score2)

            else:
                score = 0.5 * (torch.log(var2) - torch.log(var1)).sum()
        else:
            score = -(1 / (2 * torch.sqrt(self.pi_tensor) * torch.sqrt(var1))).sum()
            score += (1 / (2 * torch.sqrt(self.pi_tensor) * torch.sqrt(var2))).sum()

        if p_label1 is not None:
            if (self.use_info):
                if (self.use_kl):
                    score += (-p_label1 * torch.log(p_label1)).sum()
                    score -= (-p_label1 * torch.log(p_label2)).sum()
                else:
                    score += (-p_label1 * torch.log(p_label1)).sum()
                    score -= (-p_label2 * torch.log(p_label2)).sum()
            else:
                score += (-p_label1 * p_label1).sum()
                score -= (-p_label2 * p_label2).sum()

        return score

    def load_json(self, json_string):
        # existing load_json logic ...
        # after full tree built:
        self._build_node_map()

    # In fitting or node creation methods (e.g., cobweb, merge, split),
    # ensure new nodes are added to node_map, e.g.: after creating `new = CobwebTorchNode(...)`:
    #     self.node_map[new.id] = new

    def analyze_structure(self):
        """
        Analyze the structure of the tree:
        - Print total number of leaf nodes.
        - Print the number of nodes at each depth level (via BFS).
        - Print a histogram of number of children per parent node.
        """
        from collections import deque, defaultdict

        leaf_count = 0
        level_counts = defaultdict(int)
        child_histogram = defaultdict(int)

        queue = deque([(self.root, 0)])

        while queue:
            node, level = queue.popleft()
            level_counts[level] += 1

            if not node.children:
                leaf_count += 1
            else:
                num_children = len(node.children)
                child_histogram[num_children] += 1
                for child in node.children:
                    queue.append((child, level + 1))

        print(f"\nTotal number of leaf nodes: {leaf_count}\n")

        print("Number of nodes at each level:")
        for level in sorted(level_counts.keys()):
            print(f"  Level {level}: {level_counts[level]} node(s)")

        print("\nParent nodes by number of children:")
        for num_children in sorted(child_histogram.keys()):
            print(f"  {child_histogram[num_children]} parent(s) with {num_children} child(ren)")
