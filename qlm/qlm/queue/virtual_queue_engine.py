from bidict import bidict
import uuid
from qlm.queue.virtual_queue import VirtualQueue
from qlm.queue.group import Group
from qlm.queue.request import Request
from qlm.scheduler.scheduler import Scheduler
import random
from collections import deque, defaultdict
import time
import numpy as np
import os
import heapq
from typing import List, Tuple, Any

class VirtualQueueEngine:
    """
    VirtualQueueEngine is the main class that manages the virtual queues and groups.
    """

    def __init__(self):
        """
        Initializes the VirtualQueueEngine with empty virtual queues, request to group mapping, group to virtual queue
        mapping, virtual queue to worker mapping, model-slo to group mapping and a scheduler.
        """
        self.vqs = []
        self.request_to_group = {}
        self.group_to_vq = {}
        self.vq_worker_bimap = bidict({})
        self.model_slo_group_bimap = bidict({})
        self.scheduler = Scheduler()
        # --- EDF sort strategy (benchmark toggle) ---
        self.edf_sort_algo = os.getenv("QLM_SORT_ALGO", "timsort").lower()
        self.edf_sort_profile = os.getenv("QLM_SORT_PROFILE", "0") == "1"
        self._sort_calls = 0
        self._sort_stats = defaultdict(float)  # total_s, max_s
        self._sort_last_report = time.time()

    @staticmethod
    def _edf_deadline_of_group(g: Group) -> float:
        # (1 Request = 1 Group) 가정은 현재 주석과 동일
        r = g.requests[0]
        return float(r.insertion_time + r.slo)
    
    def _sort_groups_by_deadline(self, groups: List[Group], algo: str) -> List[Group]:
        n = len(groups)
        if n <= 1:
            return groups
    
        # key를 한 번만 계산 (공정 비교 + 오버헤드 절감)
        items: List[Tuple[float, int, Group]] = [
            (self._edf_deadline_of_group(g), i, g) for i, g in enumerate(groups)
        ]
    
        if algo in ("timsort", "sorted", "python"):
            items.sort(key=lambda t: t[0])
            return [g for _, _, g in items]
    
        if algo in ("heap", "heapq", "heapsort"):
            heapq.heapify(items)
            return [heapq.heappop(items)[2] for _ in range(len(items))]
    
        if algo.startswith("numpy_"):
            # numpy_quicksort / numpy_mergesort / numpy_heapsort
            kind = algo.split("_", 1)[1]
            deadlines = np.fromiter((t[0] for t in items), dtype=np.float64, count=n)
            order = np.argsort(deadlines, kind=kind)
            return [items[i][2] for i in order.tolist()]
    
        if algo in ("merge", "mergesort_py"):
            items = self._merge_sort_items(items)
            return [g for _, _, g in items]
    
        if algo in ("quick", "quicksort_py"):
            items = self._quick_sort_items(items)
            return [g for _, _, g in items]
    
        if algo in ("insertion", "insertion_sort"):
            items = self._insertion_sort_items(items)
            return [g for _, _, g in items]
    
        raise ValueError(f"Unknown QLM_SORT_ALGO: {algo}")
    
    @staticmethod
    def _insertion_sort_items(items: List[Tuple[float, int, Group]]) -> List[Tuple[float, int, Group]]:
        for i in range(1, len(items)):
            key_item = items[i]
            j = i - 1
            while j >= 0 and items[j][0] > key_item[0]:
                items[j + 1] = items[j]
                j -= 1
            items[j + 1] = key_item
        return items
    
    @staticmethod
    def _merge_sort_items(items: List[Tuple[float, int, Group]]) -> List[Tuple[float, int, Group]]:
        # bottom-up iterative mergesort (stable)
        width = 1
        n = len(items)
        tmp = items[:]
        while width < n:
            for left in range(0, n, 2 * width):
                mid = min(left + width, n)
                right = min(left + 2 * width, n)
                i, j, k = left, mid, left
                while i < mid and j < right:
                    if items[i][0] <= items[j][0]:
                        tmp[k] = items[i]; i += 1
                    else:
                        tmp[k] = items[j]; j += 1
                    k += 1
                while i < mid:
                    tmp[k] = items[i]; i += 1; k += 1
                while j < right:
                    tmp[k] = items[j]; j += 1; k += 1
            items, tmp = tmp, items
            width *= 2
        return items
    
    @staticmethod
    def _quick_sort_items(items: List[Tuple[float, int, Group]]) -> List[Tuple[float, int, Group]]:
        # iterative quicksort (not stable)
        stack = [(0, len(items) - 1)]
        while stack:
            lo, hi = stack.pop()
            if lo >= hi:
                continue
            pivot = items[(lo + hi) // 2][0]
            i, j = lo, hi
            while i <= j:
                while items[i][0] < pivot:
                    i += 1
                while items[j][0] > pivot:
                    j -= 1
                if i <= j:
                    items[i], items[j] = items[j], items[i]
                    i += 1; j -= 1
            if lo < j: stack.append((lo, j))
            if i < hi: stack.append((i, hi))
        return items

    def print_queue_status(self, max_groups_per_vq: int = 5):
        """
        디버그용: 각 VQ의 대기 그룹 수와 head deadline 등을 출력
        """
        now = time.time()
        lines = []
        for i, vq in enumerate(self.vqs):
            if len(vq.groups) == 0:
                continue

            head_g = vq.groups[0]
            head_r = head_g.requests[0]  # (현재 구현은 1 group = 1 request)
            head_deadline = head_r.insertion_time + head_r.slo
            head_slack = head_deadline - now

            # 앞쪽 몇 개 그룹 deadline 프리뷰
            preview = []
            for g in list(vq.groups)[:max_groups_per_vq]:
                r = g.requests[0]
                preview.append(r.insertion_time + r.slo)

            lines.append(
                f"[VQ{i}] groups={len(vq.groups)} "
                f"head_deadline={head_deadline:.3f} slack={head_slack:.3f} "
                f"model={head_r.model}"
            )
            lines.append("  deadlines: " + ", ".join(f"{d:.3f}" for d in preview))

        if lines:
            print("\n".join(lines))

    def add_worker(self, worker):
        """
        Adds a worker to the virtual queue engine. Creates a new virtual queue associated with the worker.
        :param worker: Worker object
        """
        new_vq = VirtualQueue()
        self.vqs.append(new_vq)
        self.vq_worker_bimap[new_vq] = worker

    def add_request(self, request):
        """
        Adds a request to the virtual queue engine. If a group with the same model and slo exists, adds the request to
        the group. Otherwise, creates a new group and adds the request to the new group.
        :param request: Request object
        """
        # 같은 (model, slo 존재 ==> Group 합침 (EDF 에서는 무의미)
        #if (request.model, request.slo) in self.model_slo_group_bimap:
        #    existing_group = self.model_slo_group_bimap[(request.model, request.slo)]
        #    existing_group.add_request(request)
        #    self.request_to_group[request] = existing_group
        #else:
        #    new_group = Group(request.model, request.slo)
        #    print("Adding new group with model and slo", request.model, request.slo)
        #    new_group.add_request(request)

        #===========================================================================================
        new_group=Group(request.model,request.slo)
        new_group.add_request(request)
        #===========================================================================================

        #    self.model_slo_group_bimap[(request.model, request.slo)] = new_group
        self.request_to_group[request] = new_group

            # Select a random virtual queue to add the group to
            #vq_idx = random.choice(range(len(self.vqs)))
        vq_idx=0
        
        if len(self.vqs)>1:
            vq_idx=random.choice(range(len(self.vqs)))
        
        self.vqs[vq_idx].add_group(new_group)

    def pop_request(self, worker):
        """
        Pops a request from the virtual queue associated with the worker. If the group is empty, pops the group from the
        virtual queue.
        :param worker: Worker object
        :return: Request object
        """
        vq = self.vq_worker_bimap.inv[worker]
        group = vq.get_head_group()
        request = group.pop_request()

        if len(group.requests) == 0:
            vq.pop_group()

        return request

    def has_request(self, worker):
        """
        Checks if the virtual queue associated with the worker has any requests.
        :param worker: Worker object
        :return: Boolean
        """
        vq = self.vq_worker_bimap.inv[worker]
        return len(vq.groups) > 0



#    def reorder_vqs(self):
#        """
#        [수정됨] Pure EDF Implementation
#        각 가상 큐 '내부'의 그룹들을 '절대 마감 시간(Deadline)' 순서로 정렬합니다.
#        """
#        for vq in self.vqs:
#            if len(vq.groups) > 0:
#                # 1. 정렬 기준: 절대 마감 시간 (도착 시간 + SLO)
#                # (1 Request = 1 Group이므로, group.requests[0]이 곧 해당 요청임)
#                
#                # 리스트로 변환하여 정렬
#                sorted_groups = sorted(
#                    list(vq.groups), 
#                    key=lambda g: g.requests[0].insertion_time + g.requests[0].slo
#                )
#                
#                # 2. 다시 deque로 변환하여 적용
#                vq.groups = deque(sorted_groups)
    def reorder_vqs(self):
        """
        [수정됨] Pure EDF Implementation
        각 가상 큐 '내부'의 그룹들을 '절대 마감 시간(Deadline)' 순서로 정렬합니다.
        """
        for vq in self.vqs:
            if len(vq.groups) > 1:
                groups = list(vq.groups)
    
                t0 = time.perf_counter()
                sorted_groups = self._sort_groups_by_deadline(groups, self.edf_sort_algo)
                dt = time.perf_counter() - t0
    
                vq.groups = deque(sorted_groups)
    
                if self.edf_sort_profile:
                    self._sort_calls += 1
                    self._sort_stats["total_s"] += dt
                    self._sort_stats["max_s"] = max(self._sort_stats["max_s"], dt)
    
                    now = time.time()
                    if now - self._sort_last_report >= 5.0:
                        avg_ms = (self._sort_stats["total_s"] / max(1, self._sort_calls)) * 1000.0
                        max_ms = self._sort_stats["max_s"] * 1000.0
                        print(f"[SORT] algo={self.edf_sort_algo} calls={self._sort_calls} avg={avg_ms:.3f}ms max={max_ms:.3f}ms")
                        self._sort_last_report = now
