#!/usr/bin/env python
# coding: utf-8

# # Cobweb for Language
# 
# ### Clustering Metaphor Explanation
# 
# When you're moving from place to place, obviously, everything needs to be past, you need to find a happy medium for packing - too many boxes, and you'll have everything neatly organized, but you'll use up too much cardboard. Too few boxes, and you'll save on cardboard, but it'll be much harder to locate things when you need them. Finding the optimal number of "boxes" is what modern clustering algorithms attempt to optimize.
# 
# Cobweb is my proposed solution to the optimal amount of "boxes", databases, to sort different text chunks in for optimal retrieval without a large waste of resources.
# 
# ### Applications
# 
# The implementation below follows from the metaphor, using a Cobweb tree of embeddings representing sentences to rerank specific queries by clustering and categorize different "subtrees" as clusters, training vector-store databases on each of them. Though vector-store databases (FAISS) are currently computed statically after generation of the tree, using a database such as Qdrant (which supports effective splitting and merging of databases in line with the four actions that Cobweb defines) could create a dynamic database.
# 
# The below model uses word vector embeddings, passing them into the CobwebTorchTree GitHub implementation. The model then finds cosine similarities between all parents and children and by a threshold, identifies which edges are strong enough to determine cluster similarity.
# 
# ### Other use-cases
# 
# We just used Cobweb for general text clustering but with greater complexities of word-embeddings / low-dimensionality latent-space, and possibly a better function than direct cosine comparison for sentence / text chunk similarity, this has a lot of potential as well - it allows us to not only use Cobweb as an ending retrieval classifier, but also as a fully-fledged search algorithm that can quickly evolve and reshape itself.

# ## Dependency Installation!
# 
# FOR SOME REASON: the visualize function is not properly working with this but lowkey the only thing that matters is the CobwebTorchTree implementation and we're reprogramming that manually as well so we're good here.
# 
# Additionally, the curr_ll variable breaks the predict function so we've copy-pasted and commented it out! (Predict function is also broken in cobweb-variations)

# In[ ]:


get_ipython().system('pip install faiss-cpu')


# In[ ]:


get_ipython().system('pip install -U datasets')
get_ipython().system('pip install pytrec_eval')
get_ipython().system('pip install ir-datasets')


# In[ ]:


get_ipython().system('pip install annoy')
get_ipython().system('pip install hnswlib')


# ## Cobweb Installation
# 
# The Cobweb Node class and CobwebTorchTree class (fixed for impl)
# 
# Also supports GREEDY_MODE which only evaluates the option of addition as a leaf, populating all the way, which is probably optimal for us!

# In[ ]:


COBWEB_GREEDY_MODE = False


# In[ ]:


import json
import math
from random import shuffle
from random import random
from math import log
from math import isclose
from collections import defaultdict
from collections import Counter
import heapq
import uuid
import time

import torch

class CobwebTorchNode(object):
    """
    A CobwebNode represents a concept within the knoweldge base of a particular
    :class:`CobwebTree`. Each node contains a probability table that can be
    used to calculate the probability of different attributes given the concept
    that the node represents.

    In general the :meth:`CobwebTree.ifit`, :meth:`CobwebTree.categorize`
    functions should be used to initially interface with the Cobweb knowledge
    base and then the returned concept can be used to calculate probabilities
    of certain attributes or determine concept labels.

    This constructor creates a CobwebNode with default values. It can also be
    used as a copy constructor to "deepcopy" a node, including all references
    to other parts of the original node's CobwebTree.

    :param otherNode: Another concept node to deepcopy.
    :type otherNode: CobwebNode
    """
    # a counter used to generate unique concept names.
    _counter = 0

    def __init__(self, shape, device=None, otherNode=None):
        """Create a new CobwebNode"""
        self.concept_id = self.gensym()
        self.count = torch.tensor(0.0, dtype=torch.float, device=device,
                                  requires_grad=False)
        self.mean = torch.zeros(shape, dtype=torch.float, device=device,
                                requires_grad=False)
        self.meanSq = torch.zeros(shape, dtype=torch.float, device=device,
                                  requires_grad=False)
        self.label_counts = torch.tensor([], dtype=torch.float, device=device,
                                         requires_grad=False)
        self.total_label_count = torch.tensor(0, dtype=torch.float,
                                              device=device,
                                              requires_grad=False)

        self.id = str(uuid.uuid4()) # A MONKEYPATCH WE MADE
        self.timestamp = time.time() # A MONKEYPATCH WE MADE

        self.children = []
        self.parent = None
        self.tree = None

        if otherNode:
            self.tree = otherNode.tree
            self.parent = otherNode.parent
            self.update_counts_from_node(otherNode)

            for child in otherNode.children:
                self.children.append(CobwebTorchNode(shape=self.tree.shape, device=self.tree.device, otherNode=child))

    def update_label_count_size(self):
        if self.label_counts.shape[0] < len(self.tree.labels):
            num_new = len(self.tree.labels) - self.label_counts.shape[0]
            new_counts = (self.tree.alpha + torch.zeros(num_new,
                                                        dtype=torch.float,
                                                        device=self.tree.device))
            self.label_counts = torch.cat((self.label_counts, new_counts))
            self.total_label_count += new_counts.sum()

    def increment_counts(self, instance, label):
        """
        Increment the counts at the current node according to the specified
        instance.

        :param instance: A new instances to incorporate into the node.
        :type instance: :ref:`Instance<instance-rep>`
        """
        self.count += 1
        delta = instance - self.mean
        self.mean += delta / self.count
        self.meanSq += delta * (instance - self.mean)

        self.update_label_count_size()

        if label is not None:
            self.label_counts[self.tree.labels[label]] += 1
            self.total_label_count += 1

    def update_counts_from_node(self, other):
        """
        Increments the counts of the current node by the amount in the
        specified node.

        This function is used as part of copying nodes and in merging nodes.

        :param node: Another node from the same CobwebTree
        :type node: CobwebNode
        """
        delta = other.mean - self.mean
        self.meanSq = (self.meanSq + other.meanSq + delta * delta *
                       ((self.count * other.count) / (self.count + other.count)))
        self.mean = ((self.count * self.mean + other.count * other.mean) /
                     (self.count + other.count))
        self.count += other.count

        self.update_label_count_size()
        other.update_label_count_size()

        # alpha is added to all the counts, so need to subtract from other to
        # not double count
        self.label_counts += (other.label_counts - self.tree.alpha)
        self.total_label_count += (other.total_label_count -
                                   other.label_counts.shape[0] * self.tree.alpha)

    @property
    def std(self):
        return torch.sqrt(self.var)

    @property
    def var(self):
        return self.tree.compute_var(self.meanSq, self.count)

    def log_prob_class_given_instance(self, instance, label):
        log_prob = self.log_prob(instance, label)
        log_prob += torch.log(self.count) - torch.log(self.tree.root.count)
        return log_prob

    def log_prob(self, instance, label):

        log_prob = 0

        var = self.var
        log_prob += -(0.5 * torch.log(var) + 0.5 * torch.log(2 * self.tree.pi_tensor) +
                     0.5 * torch.square(instance - self.mean) / var).sum()

        self.update_label_count_size()
        if label is not None:
            if label not in self.tree.labels:
                log_prob += (torch.log(self.tree.alpha) -
                             torch.log(self.total_label_count +
                                       self.tree.alpha))
            elif self.total_label_count > 0:
                log_prob += (torch.log(self.label_counts[self.tree.labels[label]]) -
                             torch.log(self.total_label_count))

        return log_prob

    def score_insert(self, instance, label):
        """
        Returns the score that would result from inserting the instance into
        the current node.

        This operation can be used instead of inplace and copying because it
        only looks at the attr values used in the instance and reduces iteration.
        """
        count = self.count + 1
        delta = instance - self.mean
        mean = self.mean + delta / count
        meanSq = self.meanSq + delta * (instance - mean)

        # hopefully cheap if already updated.
        self.update_label_count_size()

        label_counts = self.label_counts.clone()
        total_label_count = self.total_label_count.clone()
        if label is not None:
            label_counts[self.tree.labels[label]] += 1
            total_label_count += 1

        var = self.tree.compute_var(meanSq, count)

        std = torch.sqrt(var)

        if (self.tree.use_info):
            # score = (0.5 * torch.log(2 * self.tree.pi_tensor * var) + 0.5).sum()
            score = (0.25 * torch.log(var)).sum()
        else:
            score = -(1 / (2 * torch.sqrt(self.tree.pi_tensor) * std)).sum()

        if self.total_label_count > 0:
            p_label = label_counts / total_label_count
            if (self.tree.use_info):
                score += (-p_label * torch.log(p_label)).sum()
            else:
                score += (-p_label * p_label).sum()

        return score

    def score_merge(self, other, instance, label):
        """
        Returns the expected correct guesses that would result from merging the
        current concept with the other concept and inserting the instance.

        This operation can be used instead of inplace and copying because it
        only looks at the attr values used in the instance and reduces iteration.
        """
        delta = other.mean - self.mean
        meanSq = (self.meanSq + other.meanSq + delta * delta *
                  ((self.count * other.count) / (self.count + other.count)))
        mean = ((self.count * self.mean + other.count * other.mean) /
                (self.count + other.count))
        count = self.count + other.count

        count = count + 1
        delta = instance - mean
        mean += delta / count
        meanSq += delta * (instance - mean)

        # hopefully cheap if already updated.
        self.update_label_count_size()
        other.update_label_count_size()

        label_counts = self.label_counts.clone()
        total_label_count = self.total_label_count.clone()

        label_counts += (other.label_counts - self.tree.alpha)
        total_label_count += (other.total_label_count -
                              other.label_counts.shape[0] * self.tree.alpha)

        if label is not None:
            label_counts[self.tree.labels[label]] += 1
            total_label_count += 1

        var = self.tree.compute_var(meanSq, count)

        if (self.tree.use_info):
            # score = (0.5 * torch.log(2 * self.tree.pi_tensor * var) + 0.5).sum()
            score = (0.5 * torch.log(var)).sum()
        else:
            score = -(1 / (2 * torch.sqrt(self.tree.pi_tensor) * torch.sqrt(var))).sum()

        if self.total_label_count > 0:
            p_label = label_counts / total_label_count
            if (self.tree.use_info):
                score += (-p_label * torch.log(p_label)).sum()
            else:
                score += (-p_label * p_label).sum()

        return score

    def get_basic(self):
        """
        Climbs up the tree from the current node (probably a leaf),
        computes the category utility score, and returns the node with
        the highest score.
        """
        curr = self
        best = self
        best_cu = self.category_utility()

        while curr.parent:
            curr = curr.parent
            curr_cu = curr.category_utility()
            if curr_cu > best_cu:
                best = curr
                best_cu = curr_cu

        return best

    def get_best(self, instance, label=None):
        """
        Climbs up the tree from the current node (probably a leaf),
        computes the category utility score, and returns the node with
        the highest score.
        """
        curr = self
        best = self
        best_ll = self.log_prob_class_given_instance(instance, label)

        while curr.parent:
            curr = curr.parent
            curr_ll = curr.log_prob_class_given_instance(instance, label)
            if curr_ll > best_ll:
                best = curr
                best_ll = curr_ll

        return best

    def category_utility(self):
        p_of_c = self.count / self.tree.root.count
        root_mean, root_var, root_p_label = self.tree.root.mean_var_plabel()
        curr_mean, curr_var, curr_p_label = self.mean_var_plabel()

        return p_of_c * self.tree.compute_score(curr_mean, curr_var,
                                                curr_p_label, root_mean,
                                                root_var, root_p_label)

    def mean_var_plabel_new(self, instance, label):
        label_counts = (self.tree.alpha + torch.zeros(len(self.tree.labels),
                                                      dtype=torch.float,
                                                      device=self.tree.device))
        total_label_count = self.tree.alpha * len(self.tree.labels)

        if label is not None:
            label_counts[self.tree.labels[label]] += 1
            total_label_count += 1

        if total_label_count > 0:
            p_label = label_counts / total_label_count
        else:
            p_label = None

        var = torch.zeros(self.tree.shape, dtype=torch.float,
                          device=self.tree.device)
        var += self.tree.prior_var

        return instance, var, p_label

    def mean_var_plabel(self):
        self.update_label_count_size()
        if self.total_label_count > 0:
            p_label = self.label_counts / self.total_label_count
        else:
            p_label = None

        return self.mean, self.var, p_label

    def mean_var_plabel_insert(self, instance, label):
        count = self.count + 1
        delta = instance - self.mean
        mean = self.mean + delta / count
        meanSq = self.meanSq + delta * (instance - mean)

        # hopefully cheap if already updated.
        self.update_label_count_size()

        label_counts = self.label_counts.clone()
        total_label_count = self.total_label_count.clone()
        if label is not None:
            label_counts[self.tree.labels[label]] += 1
            total_label_count += 1

        var = self.tree.compute_var(meanSq, count)

        if total_label_count > 0:
            p_label = label_counts / total_label_count
        else:
            p_label = None

        return mean, var, p_label

    def mean_var_plabel_merge(self, other, instance, label):
        delta = other.mean - self.mean
        meanSq = (self.meanSq + other.meanSq + delta * delta *
                  ((self.count * other.count) / (self.count + other.count)))
        mean = ((self.count * self.mean + other.count * other.mean) /
                (self.count + other.count))
        count = self.count + other.count

        count = count + 1
        delta = instance - mean
        mean += delta / count
        meanSq += delta * (instance - mean)

        # hopefully cheap if already updated.
        self.update_label_count_size()
        other.update_label_count_size()

        label_counts = self.label_counts.clone()
        total_label_count = self.total_label_count.clone()

        label_counts += (other.label_counts - self.tree.alpha)
        total_label_count += (other.total_label_count -
                              other.label_counts.shape[0] * self.tree.alpha)

        if label is not None:
            label_counts[self.tree.labels[label]] += 1
            total_label_count += 1

        var = self.tree.compute_var(meanSq, count)

        if total_label_count > 0:
            p_label = label_counts / total_label_count
        else:
            p_label = None

        return mean, var, p_label

    def partition_utility(self):
        """
        Return the category utility of a particular division of a concept into
        its children.

        Category utility is always calculated in reference to a parent node and
        its own children. This is used as the heuristic to guide the concept
        formation process. Category Utility is calculated as:

        .. math::

            CU(\\{C_1, C_2, \\cdots, C_n\\}) = \\frac{1}{n} \\sum_{k=1}^n
            P(C_k) \\left[ \\sum_i \\sum_j P(A_i = V_{ij} | C_k)^2 \\right] -
            \\sum_i \\sum_j P(A_i = V_{ij})^2

        where :math:`n` is the numer of children concepts to the current node,
        :math:`P(C_k)` is the probability of a concept given the current node,
        :math:`P(A_i = V_{ij} | C_k)` is the probability of a particular
        attribute value given the concept :math:`C_k`, and :math:`P(A_i =
        V_{ij})` is the probability of a particular attribute value given the
        current node.

        In general this is used as an internal function of the cobweb algorithm
        but there may be times when it would be useful to call outside of the
        algorithm itself.

        :return: The category utility of the current node with respect to its
                 children.
        :rtype: float
        """
        if len(self.children) == 0:
            return 0.0

        score = 0.0
        parent_mean, parent_var, parent_p_label = mean_var_plabel()

        for child in self.children:
            p_of_child = child.count / self.count
            child_mean, child_var, child_p_label = mean_var_plabel()
            score += p_of_child * self.tree.compute_score(child_mean,
                                                          child_var,
                                                          child_p_label,
                                                          parent_mean,
                                                          parent_var,
                                                          parent_p_label)

        return score / len(self.children)

    def get_best_operation(self, instance, label, best1, best2, best1_pu):
        """
        Given an instance, the two best children based on category utility and
        a set of possible operations, find the operation that produces the
        highest category utility, and then return the category utility and name
        for the best operation. In the case of ties, an operator is randomly
        chosen.

        Given the following starting tree the results of the 4 standard Cobweb
        operations are shown below:

        .. image:: images/Original.png
            :width: 200px
            :align: center

        * **Best** - Categorize the instance to child with the best category
          utility. This results in a recurisve call to :meth:`cobweb
          <concept_formation.cobweb.CobwebTree.cobweb>`.

            .. image:: images/Best.png
                :width: 200px
                :align: center

        * **New** - Create a new child node to the current node and add the
          instance there. See: :meth:`create_new_child
          <concept_formation.cobweb.CobwebNode.create_new_child>`.

            .. image:: images/New.png
                :width: 200px
                :align: center

        * **Merge** - Take the two best children, create a new node as their
          mutual parent and add the instance there. See: :meth:`merge
          <concept_formation.cobweb.CobwebNode.merge>`.

            .. image:: images/Merge.png
                    :width: 200px
                    :align: center

        * **Split** - Take the best node and promote its children to be
          children of the current node and recurse on the current node. See:
          :meth:`split <concept_formation.cobweb.CobwebNode.split>`

            .. image:: images/Split.png
                :width: 200px
                :align: center

        Each operation is entertained and the resultant category utility is
        used to pick which operation to perform. The list of operations to
        entertain can be controlled with the possible_ops parameter. For
        example, when performing categorization without modifying knoweldge
        only the best and new operators are used.

        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :param best1: A tuple containing the relative cu of the best child and
            the child itself, as determined by
            :meth:`CobwebNode.two_best_children`.
        :type best1: (float, CobwebNode)
        :param best2: A tuple containing the relative cu of the second best
            child and the child itself, as determined by
            :meth:`CobwebNode.two_best_children`.
        :type best2: (float, CobwebNode)
        :param possible_ops: A list of operations from ["best", "new", "merge",
            "split"] to entertain.
        :type possible_ops: ["best", "new", "merge", "split"]
        :return: A tuple of the category utility of the best operation and the
            name of the best operation.
        :rtype: (cu_bestOp, name_bestOp)
        """
        if not best1:
            raise ValueError("Need at least one best child.")

        operations = []

        operations.append((best1_pu, random(), "best"))
        operations.append((self.pu_for_new_child(instance, label), random(), 'new'))
        if len(self.children) > 2 and best2:
            operations.append((self.pu_for_merge(best1, best2, instance, label),
                               random(), 'merge'))
        if len(best1.children) > 0:
            operations.append((self.pu_for_split(best1), random(), 'split'))

        operations.sort(reverse=True)
        # print(operations)
        best_op = (operations[0][0], operations[0][2])
        # print(best_op)
        return best_op

    def two_best_children(self, instance, label):
        """
        Calculates the category utility of inserting the instance into each of
        this node's children and returns the best two. In the event of ties
        children are sorted first by category utility, then by their size, then
        by a random value.

        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: the category utility and indices for the two best children
            (the second tuple will be ``None`` if there is only 1 child).
        :rtype: ((cu_best1,index_best1),(cu_best2,index_best2))
        """
        if len(self.children) == 0:
            raise Exception("No children!")

        relative_pus = []
        parent_mean, parent_var, parent_p_label = self.mean_var_plabel_insert(instance, label)

        for child in self.children:
            p_of_c = (child.count + 1) / (self.count + 1)
            mean, var, p_label = child.mean_var_plabel_insert(instance, label)
            score_gain = p_of_c * self.tree.compute_score(mean, var, p_label,
                                                          parent_mean,
                                                          parent_var,
                                                          parent_p_label)

            p_of_c = child.count / (self.count + 1)
            mean, var, p_label = child.mean_var_plabel()
            score_gain -= p_of_c * self.tree.compute_score(mean, var, p_label,
                                                           parent_mean,
                                                           parent_var,
                                                           parent_p_label)

            relative_pus.append((score_gain, child.count, random(), child))

        # relative_pus = [(
        #     self.pu_for_insert(child, instance, label),
        #     # child.count * child.score() - (child.count + 1) * child.score_insert(instance, label),
        #                  child.count, random(), child) for child in
        #                 self.children]

        relative_pus.sort(reverse=True)

        best1 = relative_pus[0][3]
        if COBWEB_GREEDY_MODE:
            best1_pu = 0
        else:
            best1_pu = self.pu_for_insert(best1, instance, label)

        best2 = None
        if len(relative_pus) > 1:
            best2 = relative_pus[1][3]

        return best1_pu, best1, best2

    def pu_for_insert(self, child, instance, label):
        """
        Compute the category utility of adding the instance to the specified
        child.

        This operation does not actually insert the instance into the child it
        only calculates what the result of the insertion would be. For the
        actual insertion function see: :meth:`CobwebNode.increment_counts` This
        is the function used to determine the best children for each of the
        other operations.

        :param child: a child of the current node
        :type child: CobwebNode
        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: the category utility of adding the instance to the given node
        :rtype: float

        .. seealso:: :meth:`CobwebNode.two_best_children` and
            :meth:`CobwebNode.get_best_operation`

        """
        score = 0.0
        parent_mean, parent_var, parent_p_label = self.mean_var_plabel_insert(instance, label)

        for c in self.children:
            if c == child:
                p_of_child = (c.count + 1) / (self.count + 1)
                child_mean, child_var, child_p_label = c.mean_var_plabel_insert(instance, label)
            else:
                p_of_child = (c.count) / (self.count + 1)
                child_mean, child_var, child_p_label = c.mean_var_plabel()

            score += p_of_child * self.tree.compute_score(child_mean,
                                                          child_var,
                                                          child_p_label,
                                                          parent_mean,
                                                          parent_var,
                                                          parent_p_label)

        return score / len(self.children)

    def create_new_child(self, instance, label):
        """
        Create a new child (to the current node) with the counts initialized by
        the *given instance*.

        This is the operation used for creating a new child to a node and
        adding the instance to it.

        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: The new child
        :rtype: CobwebNode
        """
        new_child = CobwebTorchNode(shape=self.tree.shape, device=self.tree.device)
        new_child.parent = self
        new_child.tree = self.tree
        new_child.increment_counts(instance, label)
        self.children.append(new_child)
        return new_child

    def pu_for_new_child(self, instance, label):
        """
        Return the category utility for creating a new child using the
        particular instance.

        This operation does not actually create the child it only calculates
        what the result of creating it would be. For the actual new function
        see: :meth:`CobwebNode.create_new_child`.

        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: the category utility of adding the instance to a new child.
        :rtype: float

        .. seealso:: :meth:`CobwebNode.get_best_operation`
        """
        score = 0.0
        parent_mean, parent_var, parent_p_label = self.mean_var_plabel_insert(instance, label)

        for c in self.children:
            p_of_child = c.count / (self.count + 1)
            child_mean, child_var, child_p_label = c.mean_var_plabel()
            score += p_of_child * self.tree.compute_score(child_mean,
                                                          child_var,
                                                          child_p_label,
                                                          parent_mean,
                                                          parent_var,
                                                          parent_p_label)

        # score for new
        p_of_child = 1.0 / (self.count + 1)
        child_mean, child_var, child_p_label = c.mean_var_plabel_new(instance, label)
        score += p_of_child * self.tree.compute_score(child_mean, child_var,
                                                      child_p_label,
                                                      parent_mean, parent_var,
                                                      parent_p_label)

        return score / (len(self.children) + 1)


    def merge(self, best1, best2):
        """
        Merge the two specified nodes.

        A merge operation introduces a new node to be the merger of the the two
        given nodes. This new node becomes a child of the current node and the
        two given nodes become children of the new node.

        :param best1: The child of the current node with the best category
            utility
        :type best1: CobwebNode
        :param best2: The child of the current node with the second best
            category utility
        :type best2: CobwebNode
        :return: The new child node that was created by the merge
        :rtype: CobwebNode
        """
        new_child = CobwebTorchNode(shape=self.tree.shape, device=self.tree.device)
        new_child.parent = self
        new_child.tree = self.tree

        new_child.update_counts_from_node(best1)
        new_child.update_counts_from_node(best2)
        best1.parent = new_child
        best2.parent = new_child
        new_child.children.append(best1)
        new_child.children.append(best2)
        self.children.remove(best1)
        self.children.remove(best2)
        self.children.append(new_child)

        return new_child

    def pu_for_merge(self, best1, best2, instance, label):
        """
        Return the category utility for merging the two best children.

        This does not actually merge the two children it only calculates what
        the result of the merge would be. For the actual merge operation see:
        :meth:`CobwebNode.merge`

        :param best1: The child of the current node with the best category
            utility
        :type best1: CobwebNode
        :param best2: The child of the current node with the second best
            category utility
        :type best2: CobwebNode
        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: The category utility that would result from merging best1 and
            best2.
        :rtype: float

        .. seealso:: :meth:`CobwebNode.get_best_operation`
        """
        score = 0.0
        parent_mean, parent_var, parent_p_label = self.mean_var_plabel_insert(instance, label)

        for c in self.children:
            if c == best1 or c == best2:
                continue

            p_of_child = c.count / (self.count + 1)
            child_mean, child_var, child_p_label = c.mean_var_plabel()
            score += p_of_child * self.tree.compute_score(child_mean,
                                                          child_var,
                                                          child_p_label,
                                                          parent_mean,
                                                          parent_var,
                                                          parent_p_label)

        p_of_child = (best1.count + best2.count + 1) / (self.count + 1)
        child_mean, child_var, child_p_label = best1.mean_var_plabel_merge(best2, instance, label)
        score += p_of_child * self.tree.compute_score(child_mean, child_var,
                                                      child_p_label,
                                                      parent_mean, parent_var,
                                                      parent_p_label)

        return score / (len(self.children) - 1)


    def split(self, best):
        """
        Split the best node and promote its children

        A split operation removes a child node and promotes its children to be
        children of the current node. Split operations result in a recursive
        call of cobweb on the current node so this function does not return
        anything.

        :param best: The child node to be split
        :type best: CobwebNode
        """
        self.children.remove(best)
        for child in best.children:
            child.parent = self
            child.tree = self.tree
            self.children.append(child)

    def pu_for_split(self, best):
        """
        Return the category utility for splitting the best child.

        This does not actually split the child it only calculates what the
        result of the split would be. For the actual split operation see:
        :meth:`CobwebNode.split`. Unlike the category utility calculations for
        the other operations split does not need the instance because splits
        trigger a recursive call on the current node.

        :param best: The child of the current node with the best category
            utility
        :type best: CobwebNode
        :return: The category utility that would result from splitting best
        :rtype: float

        .. seealso:: :meth:`CobwebNode.get_best_operation`
        """
        score = 0.0
        parent_mean, parent_var, parent_p_label = self.mean_var_plabel()

        for c in self.children:
            if c == best:
                continue
            p_of_child = c.count / self.count
            child_mean, child_var, child_p_label = c.mean_var_plabel()
            score += p_of_child * self.tree.compute_score(child_mean,
                                                          child_var,
                                                          child_p_label,
                                                          parent_mean,
                                                          parent_var,
                                                          parent_p_label)

        for c in best.children:
            p_of_child = c.count / self.count
            child_mean, child_var, child_p_label = c.mean_var_plabel()
            score += p_of_child * self.tree.compute_score(child_mean,
                                                          child_var,
                                                          child_p_label,
                                                          parent_mean,
                                                          parent_var,
                                                          parent_p_label)

        return (score / (len(self.children) - 1 + len(best.children)))

    def is_exact_match(self, instance, label):
        """
        Returns true if the concept exactly matches the instance.

        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: whether the instance perfectly matches the concept
        :rtype: boolean

        .. seealso:: :meth:`CobwebNode.get_best_operation`
        """
        self.update_label_count_size()

        if label is not None and self.total_label_count == 0:
            return False

        if label is None and self.total_label_count > 0:
            return False

        if self.total_label_count > 0:
            label_counts = self.label_counts - self.tree.alpha
            p_labels = label_counts / label_counts.sum()

            if not math.isclose(p_labels[self.tree.labels[label]].item(), 1.0):
                return False

        std = torch.sqrt(self.meanSq / self.count)
        if not torch.isclose(std, torch.zeros(std.shape,device=self.tree.device)).all():
            return False
        return torch.isclose(instance, self.mean).all()

    def __hash__(self):
        """
        The basic hash function. This hashes the concept name, which is
        generated to be unique across concepts.
        """
        return hash("CobwebNode" + str(self.concept_id))

    def gensym(self):
        """
        Generate a unique id and increment the class _counter.

        This is used to create a unique name for every concept. As long as the
        class _counter variable is never externally altered these keys will
        remain unique.

        """
        self.__class__._counter += 1
        return self.__class__._counter

    def __str__(self):

        """
        Call :meth:`CobwebNode.pretty_print`
        """
        return self.pretty_print()

    def pretty_print(self, depth=0):
        """
        Print the categorization tree

        The string formatting inserts tab characters to align child nodes of
        the same depth.

        :param depth: The current depth in the print, intended to be called
                      recursively
        :type depth: int
        :return: a formated string displaying the tree and its children
        :rtype: str
        """
        ret = str(('\t' * depth) + "|-" + str(self.mean) + ":" +
                  str(self.count) + '\n')

        for c in self.children:
            ret += c.pretty_print(depth+1)

        return ret

    def depth(self):
        """
        Returns the depth of the current node in its tree

        :return: the depth of the current node in its tree
        :rtype: int
        """
        if self.parent:
            return 1 + self.parent.depth()
        return 0

    def is_parent(self, other_concept):
        """
        Return True if this concept is a parent of other_concept

        :return: ``True`` if this concept is a parent of other_concept else
                 ``False``
        :rtype: bool
        """
        temp = other_concept
        while temp is not None:
            if temp == self:
                return True
            try:
                temp = temp.parent
            except Exception:
                print(temp)
                assert False
        return False

    def num_concepts(self):
        """
        Return the number of concepts contained below the current node in the
        tree.

        When called on the :attr:`CobwebTree.root` this is the number of nodes
        in the whole tree.

        :return: the number of concepts below this concept.
        :rtype: int
        """
        children_count = 0
        for c in self.children:
            children_count += c.num_concepts()
        return 1 + children_count

    def iterative_output_json_helper(self):
        self.update_label_count_size()
        output = {}
        output['count'] = self.count.item()
        output['mean'] = self.mean.tolist()
        output['meanSq'] = self.meanSq.tolist()
        output['label_counts'] = self.label_counts.tolist()
        return json.dumps(output)

    def iterative_output_json(self):
        output = ""

        visited = set()
        curr = self

        while curr is not None:
            if curr.concept_id not in visited:
                node_output = curr.iterative_output_json_helper()
                if len(output) > 1 and output[-1] == "}":
                    output += ", "
                output += node_output[:-1]
                output += ', "children": ['
                visited.add(curr.concept_id)

            for child in curr.children:
                if child.concept_id not in visited:
                    curr = child
                    break
            else:
                curr = curr.parent
                output += "]}"

        return output

    # TODO remove and just use above output, left for legacy use with viz.
    def output_json(self):
        return json.dumps(self.output_dict())

    def visualize(self):
        from matplotlib import pyplot as plt
        plt.imshow(self.mean.numpy())
        plt.show()

    def output_dict(self):
        """
        Outputs the categorization tree in JSON form

        :return: an object that contains all of the structural information of
                 the node and its children
        :rtype: obj
        """
        self.update_label_count_size()

        output = {}
        output['name'] = "Concept" + str(self.concept_id)
        output['size'] = self.count.item()
        output['children'] = []

        temp = {}
        temp['_category_utility'] = {"#ContinuousValue#": {'mean': self.category_utility().item(), 'std': 1, 'n': 1}}

        if self.total_label_count > 0:
            temp['label'] = {label: self.label_counts[self.tree.labels[label]].item() for label in self.tree.labels}

        for child in self.children:
            output["children"].append(child.output_dict())

        output['counts'] = temp
        output['mean'] = self.mean.tolist()
        output['meanSq'] = self.meanSq.tolist()

        return output

    def predict(self, most_likely=True):
        """
        Predict the value of an attribute, using the specified choice function
        (either the "most likely" value or a "sampled" value).

        :param attr: an attribute of an instance.
        :type attr: :ref:`Attribute<attributes>`
        :param choice_fn: a string specifying the choice function to use,
            either "most likely" or "sampled".
        :type choice_fn: a string
        :param allow_none: whether attributes not in the instance can be
            inferred to be missing. If False, then all attributes will be
            inferred with some value.
        :type allow_none: Boolean
        :return: The most likely value for the given attribute in the node's
                 probability table.
        :rtype: :ref:`Value<values>`
        """
        self.update_label_count_size()

        if most_likely:
            label = None
            if self.total_label_count > 0:
                label = self.tree.reverse_labels[self.label_counts.argmax().item()]
            return self.mean.detach().clone(), label
        else:
            label = None
            if self.total_label_count > 0:
                p_labels = label_counts / label_counts.sum()
                label = self.tree.reverse_labels[torch.multinomial(p_labels, 1).item()]
            return torch.normal(self.mean, self.std), label


# In[ ]:


import json
import math
from random import shuffle, random
from math import log, isclose
from collections import defaultdict
import heapq

import torch
from sklearn.metrics.pairwise import cosine_similarity

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

    def _cobweb_categorize(self, instance, label, use_best, greedy, max_nodes):
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

        return best if use_best else curr

    def categorize(self, instance, label=None, use_best=True, greedy=False, max_nodes=float('inf')):
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
            return self._cobweb_categorize(instance, label, use_best, greedy, max_nodes)

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


# ## Small Corpuses
# 
# Just some small corpuses to test our databases on!

# In[ ]:


import nltk
from nltk.tokenize import sent_tokenize
import numpy as np

nltk.download('punkt_tab')  # Download sentence tokenizer model

small_corpus1 = sent_tokenize("""The James Webb Space Telescope, launched in December 2021, has significantly advanced our understanding of the early universe. Its infrared capabilities allow it to peer through cosmic dust, revealing galaxies that formed over 13 billion years ago. Unlike the Hubble Telescope, which operates primarily in visible and ultraviolet light, Webb specializes in the infrared spectrum, providing complementary data that expands our astronomical knowledge.
Intermittent fasting has gained popularity in recent years as a dietary intervention with potential metabolic and cognitive benefits. Studies have suggested that time-restricted eating can improve insulin sensitivity, reduce inflammation, and potentially enhance neuroplasticity. However, the long-term effects of such eating patterns remain under active investigation, especially in diverse populations with different baseline health conditions.
The Treaty of Versailles, signed in 1919, formally ended World War I but sowed the seeds for further global conflict. By imposing harsh reparations on Germany and redrawing borders across Europe and the Middle East, the treaty inadvertently contributed to economic instability and nationalist resentment. Historians often debate whether these conditions directly facilitated the rise of authoritarian regimes in the 1930s.
Cryptocurrencies like Bitcoin and Ethereum operate on decentralized blockchain technology, enabling peer-to-peer transactions without intermediaries. While proponents argue that cryptocurrencies provide financial freedom and privacy, critics cite volatility, energy consumption, and regulatory concerns. The rise of central bank digital currencies (CBDCs) reflects a shift in how governments are responding to these innovations.
Coral bleaching is a phenomenon caused by oceanic temperature rise, often linked to climate change. When water is too warm, corals expel the algae (zooxanthellae) living in their tissues, causing them to turn completely white. While bleaching doesn't immediately kill coral, it leaves them vulnerable to disease and mortality. Global conservation efforts aim to reduce emissions and implement marine protected areas to preserve biodiversity.
James Joyce’s Ulysses is a modernist novel that chronicles a single day—June 16, 1904—in the life of Leopold Bloom. The narrative is celebrated for its stream-of-consciousness style, linguistic experimentation, and intertextual references. Although controversial upon publication, Ulysses is now regarded as a cornerstone of 20th-century literature and a seminal work in the development of the modern novel.""")

small_corpus2 = sent_tokenize("Photosynthesis is the process by which green plants convert sunlight into chemical energy. Photosynthesis primarily takes place in the chloroplasts of plant cells, using chlorophyll to absorb light. Photosynthesis involves the transformation of carbon dioxide and water into glucose and oxygen. Photosynthesis is essential for life on Earth because it provides oxygen and is the foundation of most food chains. The water cycle describes how water moves through the Earth's atmosphere, surface, and underground. The water cycle consists of processes such as evaporation, condensation, precipitation, and collection. Solar energy drives the water cycle by heating water in oceans and lakes, causing it to evaporate into the air. The water cycle plays a crucial role in regulating climate and supporting all forms of life on Earth. World War II was a global conflict that lasted from 1939 to 1945. World War II involved most of the world’s nations, forming two major opposing military alliances: the Allies and the Axis. Key events of World War II include the invasion of Poland, the Battle of Stalingrad, D-Day, and the dropping of atomic bombs on Hiroshima and Nagasaki. World War II drastically reshaped global politics and led to the formation of the United Nations. The human digestive system is responsible for breaking down food into nutrients the body can use. The human digestive system includes organs such as the mouth, esophagus, stomach, intestines, liver, and pancreas. Enzymes and digestive juices in the digestive system aid in the breakdown of carbohydrates, proteins, and fats. Nutrients are absorbed in the small intestine, while waste is excreted through the large intestine. Renewable energy comes from sources that are naturally replenished, like solar, wind, and hydro power. Renewable energy offers a sustainable alternative to fossil fuels, helping to reduce greenhouse gas emissions. Advances in technology have made renewable energy more efficient and affordable. Transitioning to renewable energy is vital for combating climate change and promoting energy security.")

small_add_corpus2 = [
    "Photosynthesis is the process by which green plants use sunlight to produce food from carbon dioxide and water.",
    "This process occurs mainly in the chloroplasts of plant cells and releases oxygen as a byproduct.",
    "World War II began in 1939 and involved most of the world’s nations, including the major powers divided into the Allies and the Axis.",
    "The war ended in 1945 after the defeat of Nazi Germany and the surrender of Japan following the atomic bombings of Hiroshima and Nagasaki."
]

user_corpus1 = sent_tokenize(
    "User prefers vegetarian recipes. "
    "User enjoys hiking on weekends. "
    "User works as a freelance graphic designer. "
    "User asks for Indian or Italian cuisine suggestions. "
    "User listens to jazz and lo-fi music. "
    "User frequently reads science fiction novels. "
    "User avoids gluten in meals. "
    "User owns a golden retriever. "
    "User likes visiting art museums. "
    "User practices yoga every morning. "
    "User uses a MacBook for creative work. "
    "User follows a low-carb diet. "
    "User is interested in learning Japanese. "
    "User commutes by bicycle. "
    "User enjoys indie films and documentaries. "
    "User plays the acoustic guitar. "
    "User volunteers at the local animal shelter. "
    "User prefers using eco-friendly products. "
    "User tracks daily habits in a bullet journal. "
    "User often asks for personal finance tips."
)

user_corpus2 = [
    "User’s name is Alex Johnson.",
    "User is 29 years old.",
    "User is male.",
    "User lives in Seattle, Washington.",
    "User works as a software engineer.",
    "User is employed at TechNova Inc.",
    "User enjoys hiking, photography, and coding.",
    "User’s favorite programming language is Python.",
    "User holds a B.S. degree in Computer Science.",
    "User graduated in 2018.",
    "User is single.",
    "User speaks English and Spanish.",
    "User has one dog named Max.",
    "User has visited 12 countries.",
    "User uses Python, JavaScript, React, and Docker.",
    "User’s GitHub username is alexj.",
    "User is passionate about technology and innovation.",
    "User often contributes to open-source projects.",
    "User values continuous learning and self-improvement.",
    "User's LOVES to eat pasta.",
    "User is gluten-free.",
    "User’s favorite food is sushi.",
]


# # FAISS with Cobweb! (DEPRECATED)
# 
# Here is a very basic TODO list: this can all get programmed INSANELY quickly if we work well and optimize.
# 
# First, a database needs to be attached to each node that isn't a sentence node but is considered a cluster (so each node with at least one child who has a sentence node). This database needs to be trained on all node-children who have sentence information.
# 
# IDEA FOR PROGRAMMING - just create and mimick the tree via FAISS databases fr - figure out if there's a database that supports easy modification in the way that the four operations of Cobweb operate.
# 
# Four operations:
# *   ADD
# *   CREATE
# *   MERGE
# *   SPLIT
# 
# Because exact FAISS lookups are not necessary for a general document search, we will be using a nearest-neighbors approach of some sort to properly optimize our approach.
# 
# We will need to modify the CobwebNode data-structure to store and modify FAISS indexes along with the node, and program custom logic for each of the four operations within the context of the database!
# 
# TODO List:
# *   Cobweb currently struggles with clustering high-text-length corpuses - possible fixes are generalizing the way that we cluster further OR improve our vectorizer
# *   Benchmarks are done, only thing left to accomplish is creating the final dynamic version of the spreadsheet (to truly utilize Cobweb).
# *   Perhaps also creating an iterative averaging process of trees to combat the fact that order determines clusters and tree structure with

# ## Static Cobweb-FAISS Database Implementation
# 
# This is the precomputed version of the thing I want to do! We're going to run some benchmarks on the pure architecture of the matter just to see exactly how good it is - promising stuff!
# 
# As aforementioned, clusters are defined by a cosine-similarity threshold to split up the Cobweb Tree into optimal subtrees. FAISS vector-store databases are then calculated on each resultant subtree.
# 
# To predict, an initial clustering assortment is done via the Cobweb Tree, and then all relevant documents are grabbed with the appropriate corresponding database - if there are not enough documents, the algorithm traverses to the nearest cluster by tree distance (as based on the way Cobweb works, that traversal produces the next most similar clusters).
# 
# The only time-sensitive problem rn is adding nodes in that it's extremely long LOL - we have to rebuild the FAISS index every single time and that's definitely not optimal in the long term!

# In[ ]:


from sentence_transformers import SentenceTransformer
import torch
from random import shuffle
import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt

class CobwebFAISSDatabase:
    def __init__(self, corpus, similarity_type="manhattan", similarity_threshold=0.75, percentile=None, verbose=False):
        # corpus: list of sentence strings
        self.sentences = list(corpus)

        self.percentile = percentile

        self.similarity_type = similarity_type

        # embed sentences
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.embeddings = self.model.encode(self.sentences, convert_to_numpy=True)
        self.embeddings_tensor = torch.tensor(self.embeddings)

        # build Cobweb tree
        self.tree = CobwebTorchTree(
            shape=self.embeddings_tensor[0].shape
            # cu_type="info",
            # covar_type="diag",
            # covar_from="root",
            # hidden_label=False,
        )
        self.min_cluster_size = 3
        self.node_to_cluster = {}
        self.sentence_to_node = {}
        self.clusters = {}
        self.cluster_vecs = {}
        self.databases = {}
        self.similarity_threshold = similarity_threshold
        self.verbose = verbose

        # fit and collect clusters
        self._fit_tree()
        self._collect_clusters()
        self._merge_small_clusters()
        self._build_faiss_indices()

    def _merge_small_clusters(self):
        min_size = self.min_cluster_size

        # Identify clusters smaller than min_size
        small_clusters = [cid for cid, sentences in self.clusters.items() if len(sentences) < min_size]

        for small_cid in small_clusters:
            # Find nodes belonging to this cluster
            nodes_in_small_cluster = [node for node, cid in self.node_to_cluster.items() if cid == small_cid]

            # Compute centroid of the small cluster
            small_centroid = np.mean(self.cluster_vecs[small_cid], axis=0)

            # Find closest other cluster by centroid distance
            closest_cid = None
            closest_dist = float('inf')

            for cid, vecs in self.cluster_vecs.items():
                if cid == small_cid:
                    continue
                centroid = np.mean(vecs, axis=0)
                dist = pairwise_distances(
                    small_centroid.reshape(1, -1), centroid.reshape(1, -1),
                    metric=self.similarity_type
                )[0, 0]
                if dist < closest_dist:
                    closest_dist = dist
                    closest_cid = cid

            if closest_cid is None:
                # Defensive: skip if no other cluster available
                continue

            # Merge small cluster sentences into closest cluster
            self.clusters[closest_cid].extend(self.clusters[small_cid])
            self.cluster_vecs[closest_cid] = np.vstack([self.cluster_vecs[closest_cid], self.cluster_vecs[small_cid]])

            # Update node to cluster mapping for all nodes in the small cluster
            for node in nodes_in_small_cluster:
                self.node_to_cluster[node] = closest_cid

            # Remove the small cluster data
            del self.clusters[small_cid]
            del self.cluster_vecs[small_cid]

    def _fit_tree(self):
        for idx, emb in enumerate(self.embeddings_tensor):
            leaf = self.tree.ifit(emb)
            if not hasattr(leaf, "sentence_ids"):
                leaf.sentence_ids = []
            leaf.sentence_ids.append(idx)
            self.sentence_to_node[idx] = leaf

    def _collect_clusters(self):
        """
        Adaptively cluster tree nodes into subtrees based on pairwise similarity threshold.
        """
        self.clusters.clear()
        self.cluster_vecs.clear()
        self.node_to_cluster = {}  # NEW: cache node -> cluster_id mapping
        visited_nodes = set()

        root = self.tree.root
        if not root:
            return

        def gather_leaves(node):
            leaves = []
            if hasattr(node, "sentence_ids"):
                leaves.append(node)
            for child in getattr(node, "children", []):
                leaves.extend(gather_leaves(child))
            return leaves

        def compute_similarity_threshold(root):
            sims = []
            queue = [root]
            while queue:
                node = queue.pop()
                for child in getattr(node, "children", []):
                    p = node.mean.numpy() if hasattr(node.mean, "numpy") else node.mean
                    c = child.mean.numpy() if hasattr(child.mean, "numpy") else child.mean
                    sim = pairwise_distances(
                        p.reshape(1, -1), c.reshape(1, -1), metric=self.similarity_type
                    )[0, 0]
                    sims.append(sim)
                    queue.append(child)

            if self.percentile:
                return np.percentile(sims, self.percentile) if sims else self.similarity_threshold
            elif self.similarity_threshold == "auto":
                if not sims:
                    return 0.75  # default fallback

                # Example auto heuristic: mean + 0.5 * std
                sims = np.array(sims)
                auto_threshold = np.mean(sims) - 2 * np.std(sims)
                if self.verbose:
                    print(f"Auto-calculated threshold: mean = {np.mean(sims):.4f}, std = {np.std(sims):.4f}, threshold = {auto_threshold:.4f}")
                return auto_threshold
            else:
                return self.similarity_threshold


        def cluster_subtree(node, visited_nodes, threshold):
            cluster_nodes = []
            def recurse(current_node):
                if current_node in visited_nodes:
                    return
                visited_nodes.add(current_node)
                cluster_nodes.append(current_node)
                for child in getattr(current_node, "children", []):
                    p = current_node.mean.numpy() if hasattr(current_node.mean, "numpy") else current_node.mean
                    c = child.mean.numpy() if hasattr(child.mean, "numpy") else child.mean
                    cos_sim = pairwise_distances(p.reshape(1, -1), c.reshape(1, -1), metric=self.similarity_type)[0, 0]
                    if self.verbose:
                        print(f"Checking edge Parent {current_node.id} ⇔ Child {child.id}: cosine = {cos_sim:.4f} (≥ {threshold:.4f})")
                    if cos_sim >= threshold:
                        recurse(child)
            recurse(node)
            return cluster_nodes

        adaptive_threshold = compute_similarity_threshold(root)
        if self.verbose:
            print(f"Using adaptive cosine similarity threshold = {adaptive_threshold} (percentile = {self.percentile})")

        cluster_id = 0
        for child in getattr(root, "children", []):
            nodes_in_cluster = cluster_subtree(child, visited_nodes, adaptive_threshold)
            sentence_ids = []
            for node in nodes_in_cluster:
                if hasattr(node, "sentence_ids"):
                    sentence_ids.extend(node.sentence_ids)

            if sentence_ids:
                cluster_key = f"cluster_{cluster_id}"
                self.clusters[cluster_key] = [self.sentences[i] for i in sentence_ids]
                self.cluster_vecs[cluster_key] = np.array([self.embeddings[i] for i in sentence_ids])
                for node in nodes_in_cluster:
                    self.node_to_cluster[node] = cluster_key  # ✅ cache mapping
                cluster_id += 1

        def assign_singletons(node):
            nonlocal cluster_id

            def find_best_cluster_along_tree(singleton_node):
                print(self.sentences[singleton_node.sentence_ids[0]])
                current = singleton_node
                best_cluster = None
                best_score = -float('inf')
                path_score = 0.0

                # Step 1: Walk up the tree to find the nearest ancestor with a cluster
                while current:
                    parent = getattr(current, "parent", None)
                    if parent is None:
                        break

                    p_vec = parent.mean.numpy() if hasattr(parent.mean, "numpy") else parent.mean
                    c_vec = current.mean.numpy() if hasattr(current.mean, "numpy") else current.mean
                    sim = -pairwise_distances(p_vec.reshape(1, -1), c_vec.reshape(1, -1), metric=self.similarity_type)[0, 0]  # negative distance = higher similarity
                    path_score += sim

                    parent_cluster = self.node_to_cluster.get(parent)
                    if parent_cluster:
                        best_cluster = parent_cluster
                        best_score = path_score
                        break

                    current = parent

                # Step 2: If no cluster was found up the tree, fallback to nearest cluster by vector distance
                if best_cluster is None:
                    query_vec = singleton_node.mean.numpy() if hasattr(singleton_node.mean, "numpy") else singleton_node.mean
                    min_dist = float('inf')

                    for cluster_id, vecs in self.cluster_vecs.items():
                        if vecs.shape[0] == 0:
                            continue
                        centroid = np.mean(vecs, axis=0)
                        dist = pairwise_distances(query_vec.reshape(1, -1), centroid.reshape(1, -1), metric=self.similarity_type)[0, 0]
                        if dist < min_dist:
                            min_dist = dist
                            best_cluster = cluster_id
                            best_score = -dist  # higher = better

                return best_cluster, best_score

            if node not in visited_nodes and hasattr(node, "sentence_ids"):
                visited_nodes.add(node)
                sentence_ids = node.sentence_ids

                if self.verbose:
                    print(f"Singleton node {node.id} with sentence_ids = {sentence_ids}")

                if sentence_ids:
                    best_cluster, best_score = find_best_cluster_along_tree(node)

                    if best_cluster is not None:
                        self.clusters[best_cluster].extend([self.sentences[i] for i in sentence_ids])
                        self.cluster_vecs[best_cluster] = np.vstack([
                            self.cluster_vecs[best_cluster],
                            np.array([self.embeddings[i] for i in sentence_ids])
                        ])
                        self.node_to_cluster[node] = best_cluster  # ✅ cache mapping
                        if self.verbose:
                            print(f"Assigned singleton node {node.id} to nearby cluster '{best_cluster}' (path score = {best_score:.4f})")
                    else:
                        cluster_key = f"cluster_{cluster_id}"
                        self.clusters[cluster_key] = [self.sentences[i] for i in sentence_ids]
                        self.cluster_vecs[cluster_key] = np.array([self.embeddings[i] for i in sentence_ids])
                        self.node_to_cluster[node] = cluster_key  # ✅ cache mapping
                        if self.verbose:
                            print(f"Created fallback singleton cluster '{cluster_key}' for node {node.id}")
                        cluster_id += 1

            for child in getattr(node, "children", []):
                assign_singletons(child)

        assign_singletons(root)

    def _build_faiss_indices(self):
        self.databases.clear()
        for cluster_id, vecs in self.cluster_vecs.items():
            if vecs.shape[0] == 0:
                continue
            index = faiss.IndexFlatL2(vecs.shape[1])
            index.add(vecs.astype(np.float32))
            self.databases[cluster_id] = index

    def _collect_nearest_clusters(self, leaf_node, input_embedding, k):
        visited_clusters = set()
        candidates = []
        dists = []

        # Helper: find the cluster this node's sentence is in
        def find_cluster_for_node(node):
            return self.node_to_cluster.get(node, None)

        # Search a specific cluster
        def search_cluster(cluster_id):
            nonlocal candidates, dists
            visited_clusters.add(cluster_id)
            vecs = self.cluster_vecs[cluster_id]
            if vecs.shape[0] == 0:
                return
            D, I = self.databases[cluster_id].search(input_embedding.astype(np.float32), k)
            for dist, idx in zip(D[0], I[0]):
                if idx != -1:
                    candidates.append(self.clusters[cluster_id][idx])
                    dists.append(dist)

        # Step 1: Search the current cluster
        current_cluster = find_cluster_for_node(leaf_node)
        if current_cluster:
            search_cluster(current_cluster)

        # Step 2: If needed, find closest additional clusters (by centroid distance)
        if len(candidates) < k:
            # Compute centroid of input query
            query_vec = input_embedding.reshape(1, -1)

            # Compute distances to all other clusters
            remaining_clusters = [
                (cluster_id, np.mean(self.cluster_vecs[cluster_id], axis=0))
                for cluster_id in self.clusters
                if cluster_id not in visited_clusters
            ]

            # Sort remaining clusters by similarity to query
            cluster_dists = [
                (cluster_id, pairwise_distances(query_vec, centroid.reshape(1, -1), metric=self.similarity_type)[0, 0])
                for cluster_id, centroid in remaining_clusters
            ]
            cluster_dists.sort(key=lambda x: x[1])  # smaller distance = more similar

            for cluster_id, _ in cluster_dists:
                search_cluster(cluster_id)
                if len(candidates) >= k:
                    break

        # Step 3: Sort all results and return top-k
        combined = sorted(zip(dists, candidates), key=lambda x: x[0])
        return [sent for _, sent in combined[:k]]


    def add_sentences(self, new_sentences: list[str]):
        """
        Add new sentences to the CobwebFAISSDatabase, updating all necessary structures:
        embeddings, tree, clusters, and FAISS indices.
        """
        if not new_sentences:
            return

        # Encode and append new sentences
        new_embeddings = self.model.encode(new_sentences, convert_to_numpy=True)
        new_embeddings_tensor = torch.tensor(new_embeddings)

        start_idx = len(self.sentences)
        self.sentences.extend(new_sentences)
        self.embeddings = np.vstack([self.embeddings, new_embeddings])
        self.embeddings_tensor = torch.cat([self.embeddings_tensor, new_embeddings_tensor], dim=0)

        # Insert new embeddings into the Cobweb tree
        for i, emb in enumerate(new_embeddings_tensor):
            idx = start_idx + i
            leaf = self.tree.ifit(emb)
            if not hasattr(leaf, "sentence_ids"):
                leaf.sentence_ids = []
            leaf.sentence_ids.append(idx)
            self.sentence_to_node[idx] = leaf

        # Rebuild clusters and FAISS indices
        self._collect_clusters()
        self._merge_small_clusters()
        self._build_faiss_indices()


    def predict(self, input_sentence, k=3, verbose=False):
        emb = self.model.encode([input_sentence], convert_to_numpy=True)
        tensor = torch.tensor(emb[0])
        leaf = self.tree.categorize(tensor, use_best=True)
        print(leaf.id)
        top = self._collect_nearest_clusters(leaf, emb, k)
        if verbose:
            print(f"Top {len(top)} results (cluster-based):")
            for s in top:
                print(" -", s)
        return top

    def print_clusters(self):
        print(f"\n{len(self.clusters)} Clusters:")
        for cluster_id, sentences in self.clusters.items():
            print(f"\nCluster {cluster_id}:")
            for sentence in sentences:
                print(f"  - {sentence}")

    def print_tree(self):
        def _print_node(node, depth=0):
            indent = "  " * depth
            label = f"{len(getattr(node, 'sentence_ids', []))} sentence(s)" if hasattr(node, "sentence_ids") else ""
            print(f"{indent}- Node ID {node.id} {label}")

            if hasattr(node, "sentence_ids"):
                for idx in node.sentence_ids:
                    print(f"{indent}    \"{self.sentences[idx]}\"")

            for child in getattr(node, "children", []):
                _print_node(child, depth + 1)

        print("\nSentence clustering tree:")
        _print_node(self.tree.root)


# ### Static Cobweb Database Basic Trials
# 
# Just some code to test the use-cases of the Cobweb database and confirm that everything works as according to plan!

# In[ ]:


corpus_to_test = user_corpus4

db = CobwebFAISSDatabase(corpus_to_test, similarity_type="manhattan", similarity_threshold="auto", percentile=None, verbose=True) # Make verbose equal to true to see cosine comparison checks

print()
print("Length of Corpus: ", len(corpus_to_test))

db.print_tree() # Uncomment this to see an indented-list version of the tree
db.print_clusters()


# In[ ]:


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances

def print_edge_similarities(node, metric, depth=0):
    """
    Recursively prints the pairwise similarity between each child and its parent.

    GPTed this to look at the pairwise similarities for the tree
    """
    for child in getattr(node, "children", []):
        # extract vectors
        p = node.mean.numpy() if hasattr(node.mean, "numpy") else node.mean
        c = child.mean.numpy() if hasattr(child.mean, "numpy") else child.mean
        # cos = cosine_similarity(p.reshape(1, -1), c.reshape(1, -1))[0, 0]
        cos = pairwise_distances(p.reshape(1, -1), c.reshape(1, -1), metric=metric)[0][0]
        indent = "  " * depth
        print(f"{indent}- Parent {node.id} ⇔ Child {child.id}: cosine = {cos:.4f}")
        print_edge_similarities(child, metric, depth + 1)

# Cosine sims along every edge:
print_edge_similarities(db.tree.root, metric='manhattan')


# In[ ]:


db.add_sentences([
    "The user has a dog named Charles.",
    "The user has a pet parrot named Parakeet.",
    "The user has over 100 pets.",
    "The user does not like taking care of his pets."
])

db.print_tree()
db.print_clusters()


# In[ ]:


test_snt = "The user is gluten-free."

db.predict(test_snt, k=5, verbose=True)


# ## Dynamic Cobweb Database Implementation
# 
# This will be the dynamic version, and will use Qdrant instead of FAISS because it supports an efficient removal, addition, etc. to enable gradually evolving and speedy additions!
# 
# We'll need to encode the databases along with the node (can do it similar to our monkey-patching idea) and then find ways to smartly merge the database.
# 
# OK so we don't want to have to rebuild every single database so this becomes very very relevant to employ!
# 
# NOTE: initially stopped the progress of this because we were interested in the CobwebAsADatabase approach but it remains to be seen whether cobweb is strong enough to be a dense classifier of text.
# 

# In[ ]:





# # Cobweb-As-A-Database Implementation
# 
# Rather than iterating over different things and assigning clusters and FAISS-indexing (probably a better method due to caching), why don't we simplify things by using Cobweb the whole way through?
# 
# Let's iterate and create this!

# ## Loading Sentence Transformer
# 
# We've gone with all-roberta-large-v1, which motivates large and comprehensive embeddings capable of generalization and specification.

# In[ ]:


from sentence_transformers import SentenceTransformer

st_model = SentenceTransformer('all-roberta-large-v1', trust_remote_code=True)


# ## Whitening Sentence Embeddings
# 
# This is a thing that I'm testing right now to make independent sentence embeddings and see the effects!

# ### Loading Sample Data!
# 
# Initially used some blank GPT-generated user statements but would like a broader sample pool to generate whitened embeddings with!
# 
# Also using:
# *   STS Dataset
# *   MS-MARCO Dataset

# In[ ]:


### Example User Corpus for our intents and purposes!

with open("user_facts_10000_mixed.txt", "r") as f:
    whitening_train_corpus = [x.strip() for x in f.readlines()]

# with open("user_questions.txt", "r") as f:
#     whitening_train_corpus += [x.strip() for x in f.readlines()]


# In[ ]:


### Pretrained Whitening Model with PCA!
convo_embs = st_model.encode(whitening_train_corpus, convert_to_numpy=True, batch_size=100, show_progress_bar=True) # took about forty minutes of training

np.save("convo_emb.npy", st_model_emb)


# In[ ]:


convo_embs = np.load("convo_emb.npy")


# In[ ]:


import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def load_sts_embeddings(model_name='all-roberta-large-v1', split='train', score_threshold=None):
    # Load STS dataset
    dataset = load_dataset("stsb_multi_mt", name="en", split=split)

    # Load sentence transformer model
    st_model = SentenceTransformer(model_name)

    embeddings = []
    labels = []

    print(f"Processing STS {split} split...")
    for item in tqdm(dataset):
        s1 = item['sentence1']
        s2 = item['sentence2']
        score = item['similarity_score'] / 5.0  # Normalize to [0, 1]

        # Optional: Only use highly similar pairs (e.g., for VAE focusing on fine semantics)
        if score_threshold is not None and score < score_threshold:
            continue

        # Get embeddings
        emb1 = st_model.encode(s1, convert_to_numpy=True)
        emb2 = st_model.encode(s2, convert_to_numpy=True)

        # Option 1: Use both individually
        embeddings.append(emb1)
        embeddings.append(emb2)

        # Option 2 (alt): Use difference, mean, or concat if you prefer contrastive-style input
        # embeddings.append(np.abs(emb1 - emb2))

        labels.append(score)
        labels.append(score)  # Both emb1 and emb2 share the score (if used individually)

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

    return embeddings, labels

sts_embeddings, sts_labels = load_sts_embeddings(score_threshold=0.8)
np.save("sts_embeddings.npy", sts_embeddings)
np.save("sts_labels.npy", sts_labels)


# In[ ]:


sts_embeddings = np.load("sts_embeddings.npy")
sts_embeddings.shape


# In[ ]:


from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import nltk
from nltk import sent_tokenize
import numpy as np
from tqdm import tqdm

nltk.download('punkt')

def build_sentence_level_msmarco(
    max_triplets=50000
):
    """
    Loads MS MARCO triplets, splits passages into sentences, and generates embeddings.
    """
    dataset = load_dataset("sentence-transformers/msmarco", "triplets", split="train")
    sentences = []
    for i, ex in enumerate(dataset):
        sentences.append(ex["query"])
        for p in [ex["positive"], ex["negative"]]:
            for sent in sent_tokenize(p):
                sentences.append(sent)
        if max_triplets and i + 1 >= max_triplets:
            break

    print(f"Total sentences: {len(sentences)}")

    embeddings = st_model.encode(sentences, convert_to_numpy=True, show_progress_bar=True)
    return np.array(embeddings), sentences

# Usage:
embs, _ = build_sentence_level_msmarco(max_triplets=25000)
np.save("msmarco_sent_embs.npy", embs)


# In[ ]:


msmarco_embs = np.load("msmarco_sent_corpus_embs.npy")
msmarco_embs.shape


# In[ ]:


wiki_embs = np.load("wiki_sentences_embeddings.npy")
wiki_embs.shape


# ### PCA + ICA Implementation
# 
# Applies a PCA to the input space and then an ICA to doubly refine the data!

# In[ ]:


import numpy as np
import pickle
from sklearn.decomposition import PCA, FastICA

class PCAICAWhiteningModel:
    def __init__(self, mean: np.ndarray, pca_components: np.ndarray, ica_unmixing: np.ndarray,
                 pca_explained_var: np.ndarray, eps: float = 1e-8):
        self.mean = mean
        self.pca_components = pca_components
        self.pca_explained_var = pca_explained_var
        self.ica_unmixing = ica_unmixing
        self.eps = eps

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Apply PCA + ICA whitening to a single embedding or a batch.
        """
        is_single = (x.ndim == 1)
        if is_single:
            x = x[np.newaxis, :]

        # Step 1: Center
        x_centered = x - self.mean

        # Step 2: PCA projection
        x_pca = np.dot(x_centered, self.pca_components.T)
        x_pca /= np.sqrt(self.pca_explained_var + self.eps)

        # Step 3: ICA transform
        x_ica = np.dot(x_pca, self.ica_unmixing.T)

        return x_ica[0] if is_single else x_ica

    @classmethod
    def fit(cls, X: np.ndarray, pca_dim: int = 256, eps: float = 1e-8,
            ica_max_iter: int = 5000, ica_tol: float = 1e-3):
        """
        Fit PCA → ICA whitening on embedding matrix X.
        """
        mean = X.mean(axis=0)
        X_centered = X - mean

        # Step 1: PCA
        pca = PCA(n_components=pca_dim)
        X_pca = pca.fit_transform(X_centered)
        components = pca.components_
        explained_var = pca.explained_variance_

        # Step 2: Normalize PCA output
        X_pca_normalized = X_pca / np.sqrt(explained_var + eps)

        # Step 3: ICA
        ica = FastICA(n_components=pca_dim, whiten='unit-variance',
                      max_iter=ica_max_iter, tol=ica_tol, random_state=42)
        X_ica = ica.fit_transform(X_pca_normalized)

        return cls(mean, components, ica.components_, explained_var, eps)

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'mean': self.mean,
                'pca_components': self.pca_components,
                'pca_explained_var': self.pca_explained_var,
                'ica_unmixing': self.ica_unmixing,
                'eps': self.eps
            }, f)

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return cls(
            mean=data['mean'],
            pca_components=data['pca_components'],
            pca_explained_var=data['pca_explained_var'],
            ica_unmixing=data['ica_unmixing'],
            eps=data['eps']
        )


# In[ ]:


EMB_DIM = 256

whitening_transform_model = PCAICAWhiteningModel.fit(np.concatenate((sts_embeddings, convo_embs), axis=0), EMB_DIM)


# In[ ]:


def encode_and_whiten_pcaica(obj) -> np.ndarray:
    """
    Encode sentences with SentenceTransformer and whiten embeddings with PCA whitening model.

    Applies PCA if a numpy array is passed in, hence general keywording!
    """
    if type(obj[0]) == str:
        # Step 1: Encode sentences to embeddings (numpy array)
        embeddings = st_model.encode(obj, convert_to_numpy=True, batch_size=64, show_progress_bar=False)

    else:
        embeddings = obj

    # Step 2: Apply whitening transform
    whitened_embeddings = whitening_transform_model.transform(embeddings)

    return whitened_embeddings


# ### Autoencoder Implementation

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer

class WhiteningAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon

def whitening_loss(z: torch.Tensor, recon: torch.Tensor, original: torch.Tensor, lambda_corr: float = 1.0):
    # Reconstruction loss
    loss_recon = F.mse_loss(recon, original)

    # Mean-centering
    z_centered = z - z.mean(dim=0, keepdim=True)

    # Covariance matrix
    cov = (z_centered.T @ z_centered) / (z_centered.size(0) - 1)

    # Deviation from identity matrix
    I = torch.eye(cov.size(0), device=z.device)
    loss_corr = F.mse_loss(cov, I)

    return loss_recon + lambda_corr * loss_corr

def train_autoencoder(embedding_array: np.ndarray, hidden_dim=256, epochs=10, batch_size=128, lr=1e-3, lambda_corr=1.0):
    input_dim = embedding_array.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WhiteningAE(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(torch.tensor(embedding_array, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            x = batch[0].to(device)
            z, recon = model(x)
            loss = whitening_loss(z, recon, x, lambda_corr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    return model


# In[ ]:


EMB_DIM = 96

embedding_data = np.concatenate((sts_embeddings, st_model_emb), axis=0)

ae_model = train_autoencoder(
    embedding_array=embedding_data,
    hidden_dim=EMB_DIM,
    epochs=20,
    batch_size=256,
    lr=1e-3,
    lambda_corr=4.0
)

def encode_sentences_ae(sentences):
    st_model.eval()
    device = next(ae_model.parameters()).device
    with torch.no_grad():
        embeddings = st_model.encode(sentences, convert_to_numpy=True)
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
        z, _ = ae_model(embeddings_tensor)
        return z.cpu().numpy()


# ### Beta Variational Autoencoder Implementation
# 
# This is BY FAR the best results we've got!!! Onto something fr, next steps are further training and diversifying the space!

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class BetaVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 256, beta: float = 4.0, dropout: float = 0.3):
        super(BetaVAE, self).__init__()
        self.beta = beta
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_dropout = nn.Dropout(dropout)
        self.decoder_fc2 = nn.Linear(hidden_dim, input_dim)

        # Skip connection weight (learnable)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # learnable blend factor

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, x_orig=None):
        h = F.relu(self.decoder_fc1(z))
        h = self.decoder_dropout(h)
        recon = self.decoder_fc2(h)

        if x_orig is not None:
            # Apply learnable skip connection
            recon = self.alpha * recon + (1 - self.alpha) * x_orig

        return recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, x_orig=x)
        return recon, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, free_bits=0.5):
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')

        # KL per dimension
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

        # Apply free bits: ignore small KL up to threshold
        kl_per_dim = torch.clamp(kl_per_dim - free_bits, min=0.0)

        # Sum over latent dims and average over batch
        kl_div = kl_per_dim.sum(dim=1).mean()

        total_loss = recon_loss + self.beta * kl_div
        return total_loss, recon_loss, kl_div


    def save_model(self, path: str):
        torch.save({
            'model_state': self.state_dict(),
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'hidden_dim': self.hidden_dim,
            'beta': self.beta
        }, path)

    @classmethod
    def load_model(cls, path: str):
        checkpoint = torch.load(path)
        model = cls(
            input_dim=checkpoint['input_dim'],
            latent_dim=checkpoint['latent_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            beta=checkpoint['beta']
        )
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        return model


# In[ ]:


from tqdm import trange
import torch
import numpy as np

def linear_beta_schedule(epoch, total_epochs, final_beta=1.0):
    return final_beta * min(1.0, epoch / total_epochs)

def cyclical_beta_schedule(epoch, cycle_length, final_beta=1.0):
    cycle_pos = epoch % cycle_length
    return final_beta * min(1.0, cycle_pos / cycle_length)

def inspect_latent_usage(model, data_tensor, batch_size=256):
    model.eval()
    z_vals = []
    with torch.no_grad():
        for i in range(0, len(data_tensor), batch_size):
            batch = data_tensor[i:i+batch_size]
            mu, _ = model.encode(batch)
            z_vals.append(mu.cpu().numpy())
    z_vals = np.concatenate(z_vals, axis=0)
    variances = np.var(z_vals, axis=0)
    print("\nLatent dimension variances:")
    relevant_dims = False
    for i, v in enumerate(variances):
        if v >= 0.01:
            relevant_dims = True
            print(f"Relevant Dim {i:02d}: {v:.6f}")

    if not relevant_dims:
        print("No relevant dimensions (all were below 0.01)")

from tqdm import trange

def train_vae(
    model,
    data,
    epochs=20,
    lr=1e-3,
    beta_schedule='constant',
    warmup_epochs=5,
    cycle_length=10,
    final_beta=10.0,
    free_bits=0.5,
    device=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)

    for epoch in trange(epochs, desc="Training", unit="epoch"):
        model.train()

        # Determine beta for this epoch
        if beta_schedule == 'linear':
            model.beta = linear_beta_schedule(epoch, warmup_epochs, final_beta)
        elif beta_schedule == 'cyclical':
            model.beta = cyclical_beta_schedule(epoch, cycle_length, final_beta)
        else:  # constant
            model.beta = final_beta

        optimizer.zero_grad()
        recon, mu, logvar = model(data_tensor)

        print(f"mu mean: {mu.mean().item():.4f}, std: {mu.std().item():.4f}")
        print(f"logvar mean: {logvar.mean().item():.4f}, std: {logvar.std().item():.4f}")

        # Pass free_bits to the loss function
        loss, recon_loss, kl_loss = model.loss_function(
            recon, data_tensor, mu, logvar, free_bits=free_bits
        )

        loss.backward()
        optimizer.step()

        print(
            f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | "
            f"Recon: {recon_loss.item():.4f} | KL: {kl_loss.item():.4f} | Beta: {model.beta:.4f}"
        )

    # Optional: inspect latent usage after training
    inspect_latent_usage(model, data_tensor)


# In[ ]:


EMB_DIM = 96
bvae_model = BetaVAE(input_dim=1024, latent_dim=EMB_DIM, hidden_dim=192, beta=0.0, dropout=0.2)

train_vae(
    bvae_model,
    np.concatenate((sts_embeddings, st_model_emb), axis=0),
    epochs=30,
    lr=5e-4,
    beta_schedule='cyclical',
    warmup_epochs=10,
    cycle_length=10,
    final_beta=4,
    free_bits=0.0,
    device=None
)


# In[ ]:


def encode_sentences_bvae(sentences):
    with torch.no_grad():
        embeddings = st_model.encode(sentences, convert_to_tensor=True)
        mu, _ = bvae_model.encode(embeddings)
    return mu.cpu().numpy()  # These are your whitened representations


# ## CAAD Implementation!
# 
# Cobweb...as a database!

# In[ ]:


from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import heapq
from queue import PriorityQueue
import itertools
import functools
from tqdm import tqdm

class CobwebAsADatabase:
    def __init__(self, corpus=None, corpus_embeddings=None, similarity_type="manhattan", encode_func=None, verbose=False):
        """
        Initializes the CobwebDatabase with an optional corpus.

        Args:
            corpus (list of str): The list of initial sentences.
            similarity_type (str): The distance metric to use ('cosine', 'euclidean', etc.).
            verbose (bool): If True, prints detailed logs.
        """
        self.encode_func = encode_func
        if not encode_func:
            self.encode_func = functools.partial(st_model.encode, convert_to_numpy=True)
        self.similarity_type = similarity_type
        self.verbose = verbose

        self.sentences = []
        self.embeddings = []
        self.sentence_to_node = {}

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.max_init_search = 50 # CONSTANT!!!

        if not encode_func:
            self.tree = CobwebTorchTree(shape=(1024,), device=self.device) # CHANGE SHAPE DEPENDING ON FINAL VALUES!
        else:
            self.tree = CobwebTorchTree(shape=(EMB_DIM,), device=self.device)

        if corpus:
            self.add_sentences(corpus, corpus_embeddings)

    def add_sentences(self, new_sentences, new_vectors=None):
        """
        Adds new sentences to the database and updates the Cobweb tree.
        """
        if new_vectors is None:
            new_embeddings = self.encode_func(new_sentences)
        else:
            if self.tree.shape[0] != new_vectors.shape[1]:
                new_embeddings = self.encode_func(new_vectors)
            else:
                new_embeddings = new_vectors
        start_index = len(self.sentences)

        for i, (sentence, emb) in tqdm(enumerate(zip(new_sentences, new_embeddings)), total=len(new_sentences), desc="Training CobwebAsADatabase"):
            self.sentences.append(sentence)
            self.embeddings.append(emb)
            leaf = self.tree.ifit(torch.tensor(emb, device=self.device))
            leaf.sentence_id = start_index + i
            self.sentence_to_node[start_index + i] = leaf

        print()

        if self.verbose:
            print(f"Added {len(new_sentences)} sentences.")

    def predict(self, input_sentence, k, sort_by_date=False, verbose=True):
        """
        Predict similar sentences using optimized DFS from Cobweb tree.
        Prioritizes children closest to input embedding.
        Sorts results by node timestamp.
        """
        emb = self.encode_func([input_sentence])
        input_vec = emb[0].reshape(1, -1)
        tensor = torch.tensor(emb[0], device=self.device)
        leaf = self.tree.categorize(tensor, use_best=True, max_nodes=self.max_init_search)

        candidates = []
        visited = set()

        def dfs(node):
            if len(candidates) >= k or node in visited:
                return
            visited.add(node)

            if hasattr(node, "sentence_id"):
                idx = node.sentence_id
                if idx is not None:
                    dist_to_input = pairwise_distances(
                        input_vec,
                        self.embeddings[idx].reshape(1, -1),
                        metric=self.similarity_type
                    )[0][0]
                    candidates.append((dist_to_input, idx, node.timestamp))
                return

            children = getattr(node, "children", [])
            if not children:
                return

            child_vecs = [child.mean.cpu().numpy().reshape(1, -1) for child in children]
            dists = pairwise_distances(input_vec, np.vstack(child_vecs), metric=self.similarity_type)[0]
            sorted_children = [child for _, child in sorted(zip(dists, children), key=lambda x: x[0])]

            for child in sorted_children:
                dfs(child)

        current = leaf
        while current is not None and len(candidates) < k:
            dfs(current)
            current = getattr(current, "parent", None)

        if verbose:
            print("Sentences ranked by order found:")
            [print("- " + s) for s in [self.sentences[idx] for _, idx, _ in sorted(heapq.nsmallest(k, candidates), key=lambda x: x[0])]]

        sort_idx = 2 if sort_by_date else 0

        top_k = sorted(heapq.nsmallest(k, candidates), key=lambda x: x[sort_idx])
        return [self.sentences[idx] for _, idx, _ in top_k]

    def predict_multiple(self, input_sentences, k, sort_by_date=False, verbose=True):
        """
        Predicts k similar documents for each input sentence.
        Avoids re-visiting already explored leaves and uses optimized DFS.
        Sorts the final output by node timestamp.
        """
        final_candidates = []
        visited_leaves = set()

        for input_sentence in input_sentences:
            emb = self.encode_func([input_sentence])
            input_vec = emb[0].reshape(1, -1)
            tensor = torch.tensor(emb[0], device=self.device)
            leaf = self.tree.categorize(tensor, use_best=True, max_nodes=self.max_init_search)

            candidates = []
            visited_nodes = set()

            def dfs_from_node(node, edge_sum):
                if node in visited_nodes or len(candidates) >= k:
                    return
                visited_nodes.add(node)

                if hasattr(node, "sentence_id") and node.sentence_id is not None and node.sentence_id not in visited_leaves:
                    idx = node.sentence_id
                    dist_to_input = pairwise_distances(
                        input_vec,
                        self.embeddings[idx].reshape(1, -1),
                        metric=self.similarity_type
                    )[0][0]
                    candidates.append((dist_to_input, node.timestamp, idx))
                    visited_leaves.add(idx)
                    return

                children = getattr(node, "children", [])
                if not children:
                    return

                parent_vec = node.mean.cpu().numpy().reshape(1, -1)
                child_vecs = [child.mean.cpu().numpy().reshape(1, -1) for child in children]
                edge_dists = pairwise_distances(input_vec, np.vstack(child_vecs), metric=self.similarity_type)[0]

                sorted_children = [child for _, child in sorted(zip(edge_dists, children), key=lambda x: x[0])]

                for child in sorted_children:
                    dfs_from_node(child, edge_sum)

            current = leaf
            while current is not None and len(candidates) < k:
                dfs_from_node(current, 0.0)
                current = getattr(current, "parent", None)

            final_candidates.extend(candidates)

        key1 = 1 if sort_by_date else 0
        key2 = 0 if sort_by_date else 1

        final_sorted = sorted(final_candidates, key=lambda x: (x[key1], x[key2]))
        return [self.sentences[idx] for _, _, idx in final_sorted]

    def print_tree(self):
        """
        Prints the structure of the Cobweb tree.
        """
        def _print_node(node, depth=0):
            indent = "  " * depth
            label = f"Sentence ID: {getattr(node, 'sentence_id', 'N/A')}"
            print(f"{indent}- Node ID {node.id} {label}")
            if hasattr(node, "sentence_id"):
                idx = node.sentence_id
                print(f"{indent}    \"{self.sentences[idx]}\"")
            for child in getattr(node, "children", []):
                _print_node(child, depth + 1)

        print("\nSentence clustering tree:")
        _print_node(self.tree.root)


# ## Minimal Working Example + comparison to FAISS

# In[ ]:


# Make sure to initialize the stuff from above!
test_corpus = user_corpus2

# new_db = CobwebAsADatabase(test_corpus, similarity_type="manhattan", encode_func=encode_and_whiten_pcaica, verbose=True)
new_db = CobwebAsADatabase(test_corpus, similarity_type="manhattan", verbose=True)
# for corp in test_corpus:
#     new_db.add_sentences([corp])

new_db.print_tree()


# In[ ]:


test_query = "What should the user have for dinner?"

print(new_db.predict(test_query, k=4, verbose=True))

# test_queries = [
#     "The user is looking for dinner options.",
#     "The user wants to explore painting."
# ]

# new_db.predict_multiple(test_queries, k=4)


# In[ ]:


import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

sentences = test_corpus

model = SentenceTransformer('all-roberta-large-v1', trust_remote_code=True)

# 3. Embed the sentences
embeddings = model.encode(sentences, convert_to_numpy=True)

# 4. Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 5. Function to query similar sentences
def search(query, top_k):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    print(f"\nQuery: {query}\nTop {top_k} similar sentences:")
    for i, idx in enumerate(indices[0]):
        print(f"{i+1}: {sentences[idx]} (distance: {distances[0][i]:.4f})")


# In[ ]:


# Example usage
search(test_query, top_k=10)


# # Benchmarks and tests!
# 
# The below tests are designed to evaluate the performative gains of Cobweb-As-A-Database in comparison to traditional industry standards (FAISS, Annoy, HNSWLib). We will attempt to evaluate four criteria through these four benchmarks!
# 
# 1. Benchmark 1 will test similarity-focused retrieval via the Quora Question Pairs Dataset, which consists of pairs of questions that are rephrased versions of each other.
# 2. Benchmark 2 will test QA-focused retrieval via the MS-Marco Dataset, which consists of questions asked on a broad corpus of general knowledge from Bing.
# 3. Benchmark 3 is still undefined (TODO NEED TO FIGURE OUT WHAT THIS IS) but has the primary goal of proving that Cobweb's method of retrieving documents is of higher accuracy than SOTA.
# 4. Benchmark 4 is perhaps the most relevant, and will detail a stress-test on repeated building and querying with a procedurally growing dataset, to prove this new
# 
# Ultimately, we should attain some mix of positive results from Benchmarks 3 and 4 to prove the efficiency of Cobweb as a retrieval algorithm. Current needs are as follows:
# 
# Benchmark 3 Requirements:
# *   Need to find a dataset that standard FAISS, ANNOY, HNSWLib perform poorly on and show that CAAD and whiteCAAD perform well on it
# 
# Benchmark 4 Requirements:
# *   Need to scale CAAD up and find a corpus for which it takes FAISS a ridiculously long time to index stuff
# 
# 

# ## Benchmark 1: Gauntlet on Quora Question Pair Dataset
# 
# We'll run a quick benchmark on the QQP Dataset to display the comparison of regular FAISS to Cobweb. We will be evaluating database architectures on the following metrics:
# 
# 
# 
# Metrics
# 
# *   Recall - How many relevant items the system retrieves out of all items retrieved, as a percentage averaged over all queries (higher = better).
# *   Mean Reciprocal Rank (MRR) - A metric that gauges how well the system ranks the returned queries, placing emphasis on order (higher = better).
# *   Normalized Discounted Cumulative Gain (NDCG) - A metric that also gauges how well the system orders the results by discounting each result's score relative to how low / high on the list it is. If good results are returned fast, we get high power output
# *   Query Latency - The speed at which the query is retrieved, measured in milliseconds (lower = better).
# 
# Architectures:
# 
# *   FAISS - stands for Facebook AI Similarity Search. Uses K-means to cluster the corpus into centroids, filtering the final search down for optimization. Most commonly used due to native support with CUDA, supporting GPU acceleration, however, it too is hard to "incrmeentally update" and requires a rebuilding of the index with new information.
# *   Spotify ANNOY - stands for Approximate Nearest Neighbors Oh Yeah. Highly optimized for read-only, memory-mapped data (so everything doesn't have to be memory-imported). Terrible to update over time, and far slower/less accurate than FAISS and HNSWLIB. This is the precise opposite of an incremental database LOL!
# *   HNSWLib - stands for Hierarchical Navigable Small World. Graph-based algorithm that tries to grow more dense as it moves from layer to layer, with the goal to optimize and cut down on the searching space. Main con is that index creation is EXTREMELY slow, and the key hyperparameters of `ef_construction` and `ef_search` are quite hard to tune. Additionally, `max_elements` is an argument of HNSWLib, meaning it can only support at most that many nodes.
# 
# Valuable Configurations to showcase in results:
# 
# GREEDY-MODE:
# *   PCAICA = 512 works for basically everything with fantastic results!
# 
# NON-GREEDY-MODE:
# *   PCAICA = 512, corpus = 2000, queries = 200, TOP_K = 3
# *   PCAICA = 256, corpus <= 1000, queries <= 400, TOP_K = 2, 3, 5
# 

# In[ ]:


EMB_DIM = 512

whitening_transform_model = PCAICAWhiteningModel.fit(np.concatenate((sts_embeddings, convo_embs), axis=0), EMB_DIM)


# In[ ]:


from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.metrics import ndcg_score
from tqdm import tqdm
import numpy as np
import faiss
import time
from random import sample as randsample
from annoy import AnnoyIndex
import hnswlib

# Load QQP dataset and extract duplicates
dataset = load_dataset("quora", split="train", trust_remote_code=True)
duplicates = [ex for ex in dataset if ex["is_duplicate"] == 1]

shuffle(duplicates)

# Sample subset for benchmarking
subset_size = 1500
sampled = randsample(duplicates, subset_size)
target_size = 200
queries = [ex["questions"]["text"][0] for ex in sampled[:target_size]]
targets = [ex["questions"]["text"][1] for ex in sampled[:target_size]]
corpus = [ex["questions"]["text"][1] for ex in sampled]

print("Length of Corpus:", len(corpus))

TOP_K = 3

# Encode corpus
print("Encoding corpus embeddings...")
start_time = time.time()
corpus_embeddings = st_model.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)
encoding_time = time.time() - start_time
print(f"Corpus embedding time: {round(encoding_time, 2)} seconds")

# === FAISS Setup ===
dim = corpus_embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dim)
start_time = time.time()
faiss_index.add(corpus_embeddings)
faiss_build_time = time.time() - start_time
print(f"FAISS index build time: {round(faiss_build_time, 2)} seconds")

def retrieve_faiss(query, k):
    query_emb = st_model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    _, ids = faiss_index.search(np.expand_dims(query_emb, axis=0), k)
    return [corpus[i] for i in ids[0]]

# === Annoy Setup ===
annoy_index = AnnoyIndex(dim, 'angular')
for i, emb in enumerate(corpus_embeddings):
    annoy_index.add_item(i, emb)
start_time = time.time()
annoy_index.build(10)
annoy_build_time = time.time() - start_time
print(f"Annoy index build time: {round(annoy_build_time, 2)} seconds")

def retrieve_annoy(query, k):
    query_emb = st_model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    ids = annoy_index.get_nns_by_vector(query_emb, k)
    return [corpus[i] for i in ids]

# === HNSWLIB Setup ===
hnsw_index = hnswlib.Index(space='cosine', dim=dim)
hnsw_index.init_index(max_elements=len(corpus), ef_construction=100, M=16)
start_time = time.time()
hnsw_index.add_items(corpus_embeddings, np.arange(len(corpus)))
hnsw_index.set_ef(50)
hnsw_build_time = time.time() - start_time
print(f"HNSWLIB index build time: {round(hnsw_build_time, 2)} seconds")

def retrieve_hnsw(query, k):
    query_emb = st_model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    ids, _ = hnsw_index.knn_query(query_emb, k=k)
    return [corpus[i] for i in ids[0]]

# === CAAD Setup (Assumes these classes exist in your env) ===
start_time = time.time()
regCAAD = CobwebAsADatabase(corpus, corpus_embeddings=corpus_embeddings, similarity_type="manhattan", verbose=False)
caad_build_time = time.time() - start_time
print(f"CAAD build time: {round(caad_build_time, 2)} seconds")

def retrieve_caad(query, k):
    return regCAAD.predict(query, k=k, verbose=False)

start_time = time.time()
whiteCAAD = CobwebAsADatabase(corpus, corpus_embeddings=corpus_embeddings, similarity_type="manhattan", encode_func=encode_and_whiten_pcaica, verbose=False)
whitecaad_build_time = time.time() - start_time
print(f"whiteCAAD build time: {round(whitecaad_build_time, 2)} seconds")

def retrieve_whitecaad(query, k):
    return whiteCAAD.predict(query, k=k, verbose=False)

# === Evaluation Function ===
def evaluate(retrieve_fn, name):
    hits, mrr_total, ndcg_total = 0, 0, 0
    latencies = []

    for query, target in tqdm(zip(queries, targets), total=len(queries), desc=name):
        start = time.time()
        retrieved = retrieve_fn(query, TOP_K)
        latencies.append(time.time() - start)

        if target in retrieved:
            hits += 1
            rank = retrieved.index(target) + 1
            mrr_total += 1 / rank

        relevance = [1 if doc == target else 0 for doc in retrieved]
        ndcg_total += ndcg_score([relevance], [list(reversed(range(len(relevance))))])

    n = len(queries)
    return {
        "method": name,
        f"recall@{TOP_K}": round(hits / n, 4),
        f"mrr@{TOP_K}": round(mrr_total / n, 4),
        f"ndcg@{TOP_K}": round(ndcg_total / n, 4),
        "avg_latency_ms": round(1000 * np.mean(latencies), 2)
    }

# === Run All Evaluations ===
faiss_results = evaluate(retrieve_faiss, "FAISS")
annoy_results = evaluate(retrieve_annoy, "Annoy")
hnsw_results = evaluate(retrieve_hnsw, "HNSWLIB")
caad_results = evaluate(retrieve_caad, "CAAD")
whitecaad_results = evaluate(retrieve_whitecaad, "whiteCAAD")

# === Print Summary ===
print(f"\n--- Benchmark Results (TOP_K={TOP_K}) ---")
for res in [faiss_results, annoy_results, hnsw_results, caad_results, whitecaad_results]:
    print(res)


# In[ ]:


regCAAD.tree.analyze_structure()

whiteCAAD.tree.analyze_structure()


# In[ ]:


start_time = time.time()
whiteCAAD = CobwebAsADatabase(corpus, corpus_embeddings=corpus_embeddings, similarity_type="manhattan", encode_func=encode_and_whiten_pcaica, verbose=False)
whitecaad_build_time = time.time() - start_time
print(f"whiteCAAD build time: {round(whitecaad_build_time, 2)} seconds")

whitecaad_results = evaluate(retrieve_whitecaad, "whiteCAAD")

print(whitecaad_results)


# In[ ]:


start_time = time.time()
x = whitening_transform_model.transform(corpus_embeddings)
whitening_overhead_time = time.time() - start_time
print(f"Whitening Overhead Time: {round(whitening_overhead_time, 2)} seconds") # proves that the PCA time is negligible, main bottleneck is with


# In[ ]:





# ## Benchmark 2: Strenous retrieval on MS-MARCO Dataset
# 
# Our first dataset basically tested direct retrieval of likeminded questions, but this dataset will be the more rigorous test on question-answer retrieval and larger-corpus retrieval!
# 
# The MS-MARCO Dataset is the industry standard for retrieval, and is an extremely comprehensive database on which most industry standard vector-stores are evaluated. For the intents and purposes of PCA+ICA, we're going to restrict eligible corpuses to under 75 words for retrieval!
# 
# Valuable Configurations to showcase in results:
# 
# GREEDY-MODE:
# *   TBD
# 
# NON-GREEDY-MODE:
# *   PCAICA = 512, distractor_size = 1500, queries = 300, TOP_K = 3
# *   PCAICA = 256 (192 through 384), corpus <= 1000, queries <= 400, TOP_K = 2, 3, 5

# In[ ]:


EMB_DIM = 512

whitening_transform_model = PCAICAWhiteningModel.fit(np.concatenate((sts_embeddings, convo_embs), axis=0), EMB_DIM)


# In[ ]:


from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.metrics import ndcg_score
from tqdm import tqdm
import numpy as np
import faiss
import time
from random import sample as randsample

# Load MS MARCO passage-ranking validation split
ds = load_dataset("ms_marco", "v2.1", split="validation")

# Extract (query, passage, is_selected) triples where passages are <= 60 words
all_passages = []
for ex in ds:
    query = ex["query"]
    passage_texts = ex["passages"]["passage_text"]
    is_selected_flags = ex["passages"]["is_selected"]

    for txt, is_sel in zip(passage_texts, is_selected_flags):
        if len(txt.split()) <= 60:
            all_passages.append((query, txt, is_sel))

# Remove duplicates (optional, but safe)
all_passages = list({(q, t, sel) for q, t, sel in all_passages})

# Collect positive pairs and distractor passages
positive_pairs = [(q, t) for q, t, sel in all_passages if sel == 1]
print(f"Total positive pairs: {len(positive_pairs)}")

# Sample ~200 positive pairs for benchmarking
sample_size = 500
sampled_pairs = randsample(positive_pairs, sample_size)
queries, targets = zip(*sampled_pairs)

# Build corpus: unique positive targets + distractors (non-relevant passages)
distractor_size = 9500
unique_targets = set(targets)
distractors = [t for _, t, sel in all_passages if sel == 0 and t not in unique_targets]
distractors_sample = randsample(distractors, min(distractor_size, len(distractors)))

corpus = list(unique_targets) + distractors_sample
print(f"Corpus size: {len(corpus)}")

TOP_K = 3

start = time.time()
corpus_embeddings = st_model.encode(corpus, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
np.save(corpus_embeddings, "ms_marco_test.npy")
from google.colab import files
files.download("ms_marco_test.npy")
corpus_time = time.time() - start

print(f"Corpus Embedding Time: {corpus_time}")

dim = corpus_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
start = time.time()
index.add(corpus_embeddings)
faiss_build = time.time() - start

def retrieve_faiss(q, k):
    emb = st_model.encode(q, convert_to_numpy=True, normalize_embeddings=True)
    _, ids = index.search(np.expand_dims(emb, 0), k)
    return [corpus[i] for i in ids[0]]

start = time.time()
regCAAD = CobwebAsADatabase(corpus, corpus_embeddings=corpus_embeddings, similarity_type="manhattan", verbose=False)
caad_build = time.time() - start

start = time.time()
whiteCAAD = CobwebAsADatabase(
    corpus,
    corpus_embeddings=corpus_embeddings,
    similarity_type="manhattan",
    encode_func=encode_and_whiten_pcaica,
    verbose=False
)
whitecaad_build = time.time() - start

def retrieve_caad(q, k):
    return regCAAD.predict(q, k=k, verbose=False)

def retrieve_whitecaad(q, k):
    return whiteCAAD.predict(q, k=k, verbose=False)

def evaluate(fn, name, build_time):
    hits = mrr = ndcg = 0
    latencies = []
    for q, t in tqdm(zip(queries, targets), total=len(queries), desc=name):
        st = time.time()
        res = fn(q, TOP_K)
        latencies.append(time.time() - st)
        if t in res:
            hits += 1
            rank = res.index(t) + 1
            mrr += 1 / rank
        rel = [1 if doc == t else 0 for doc in res]
        ndcg += ndcg_score([rel], [list(range(len(rel)-1, -1, -1))])
    n = len(queries)
    return {
        "method": name,
        f"recall@{TOP_K}": round(hits/n, 4),
        f"mrr@{TOP_K}": round(mrr/n, 4),
        f"ndcg@{TOP_K}": round(ndcg/n, 4),
        "avg_latency_ms": round(1000 * np.mean(latencies), 2),
        "build_time_s": round(build_time, 2)
    }

results = [
    evaluate(retrieve_faiss,   "FAISS",    faiss_build),
    evaluate(retrieve_caad,    "CAAD",     caad_build),
    evaluate(retrieve_whitecaad,"whiteCAAD",whitecaad_build),
]

print(f"\n--- Retrieval Benchmark (MS MARCO, 100x queries, K={TOP_K}) ---")
for r in results:
    print(r)


# ## Benchmark 3: Accuracy Test
# 
# Extremely tentatively, I'd like for our third test to be an accuracy-focused test that takes advantage of Cobweb's method of retrieving similarity. This corpus embedding situation

# In[ ]:





# ## Benchmark 4: Stress Test
# 
# Tentatively, I want our last test to be some kind of database addition test that showcases the pros of the incremental addition to the database!
# 
# One potential pathway:
# *   Add the first 100 documents universally, and then add documents and then iterate between queries and addition and measure the timing of each one!
# *   New bottleneck acquired - ifit is far too slow because it searches EVERYTHING - is there a way to only concern ourselves with the most probable outcomes?

# In[ ]:




