from qlm.queue.group import Group
from qlm.queue.virtual_queue import VirtualQueue
from qlm.config import Config
from qlm.scheduler.rwt_estimator import RWTEstimator
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum
from bidict import bidict
import time
from collections import deque
import copy

class Scheduler:
    """
    Scheduler class is responsible for managing the scheduling of the queue.
    """

    def __init__(self, policy="edf"):
        self.policy = policy
        self.rwt_estimator = RWTEstimator()
        self.config = Config()

    def _update_all_slos(self, vqs):
        for vq in vqs:
            for group in vq.groups:
                for request in group.requests:
                    curr_time = time.time()
                    request.slo -= curr_time - request.insertion_time
                    request.slo = (
                        request.slo
                        // self.config.slo_granularity
                        * self.config.slo_granularity
                    )
                    request.insertion_time = curr_time

    def check_violation(self, vqs):
        self._update_all_slos(vqs)
        for vq in vqs:
            est_time = 0
            curr_model = None
            for group in vq.groups:
                prev_model = curr_model
                curr_model = group.model
                if prev_model != None and prev_model != curr_model:
                    est_time += self.config.model_swap_time # config 접근 수정
                waiting_time = self.rwt_estimator.get_waiting_time(group)
                est_time += waiting_time
                if est_time > group.slo:
                    return True
        return False

    def reorder(self, vqs):
        if self.policy == "edf":
            return self._reorder_edf(vqs)
        elif self.policy == "lp":
            return self._reorder_lp_solver(vqs)

    def _reorder_edf(self, vqs):
        for vq in vqs:
            groups = list(vq.groups)
            groups.sort(key=lambda x: x.slo)
            vq.groups = deque(groups)
        return vqs

    def _reorder_lp_solver(self, vqs):
        """
        Reorders the virtual queues based on the Linear Programming (LP) solver.
        """
        print(f"DEBUG: Gurobi Solver Called at {time.time()}") # <--- 이 한 줄 추가
        options = {
            "WLSACCESSID": self.config.gurobi["access_id"],
            "WLSSECRET": self.config.gurobi["secret_key"],
            "LICENSEID": self.config.gurobi["license"],
            # "OutputFlag": 0, # 로그 너무 많이 뜨면 0으로 설정
        }

        groups = []
        slos = []
        models = []

        model_idx_bimap = bidict({})
        group_idx_bimap = bidict({})
        last_model_idx = 0
        last_group_idx = 0

        # [중요 변경 1] 큐를 미리 비우지(popleft) 않고 순회만 해서 정보를 수집합니다.
        # 이렇게 하면 원본 vqs가 보존됩니다.
        for vq in vqs:
            for group in vq.groups:
                groups.append(self.rwt_estimator.get_waiting_time(group))
                group_idx_bimap[group] = last_group_idx
                last_group_idx += 1
                slos.append(group.slo)
                if group.model in model_idx_bimap:
                    models.append(model_idx_bimap[group.model])
                else:
                    model_idx_bimap[group.model] = last_model_idx
                    models.append(last_model_idx)
                    last_model_idx += 1

        N = len(groups)
        WORKERs = len(vqs)
        SLOTs = len(groups)
        
        # 그룹이 하나도 없으면 바로 리턴 (Gurobi 에러 방지)
        if N == 0:
            return vqs

        # 모델 수 계산 (0개일 경우 방지)
        MODELs = len(models) if len(models) > 0 else 1
        MODEL_SWAP_TIME = self.config.model_swap_time

        try:
            with gp.Env(params=options) as env, gp.Model(env=env) as model:
                # 타임리밋 설정 (1초 안에 못 찾으면 포기)
                model.setParam('TimeLimit', 1.0) 

                # --- 변수 및 제약 조건 설정 (기존과 동일) ---
                x = model.addVars(WORKERs, N, N, vtype=GRB.BINARY, name="x")
                completion_slot = model.addVars(WORKERs, N, vtype=GRB.INTEGER, name="ct")
                model_slot = model.addVars(WORKERs, N, vtype=GRB.INTEGER, name="model")
                transition_slot = model.addVars(WORKERs, N, vtype=GRB.INTEGER, name="trans")
                slo_slot = model.addVars(WORKERs, N, vtype=GRB.INTEGER, name="slo")
                penalty_slot = model.addVars(WORKERs, N, vtype=GRB.INTEGER, name="penalty", lb=-100000)

                for i in range(N):
                    model.addConstr(quicksum(x[g, i, slot] for g in range(WORKERs) for slot in range(SLOTs)) == 1)

                for g in range(WORKERs):
                    for slot in range(SLOTs):
                        model.addConstr(quicksum(x[g, i, slot] for i in range(N)) == 1)

                for g in range(WORKERs):
                    for slot in range(SLOTs):
                        model.addConstr(model_slot[g, slot] == quicksum(models[i] * x[g, i, slot] for i in range(N)))
                        model.addConstr(slo_slot[g, slot] == quicksum(slos[i] * x[g, i, slot] for i in range(N)))

                for g in range(WORKERs):
                    model.addConstr(transition_slot[g, 0] == 0)

                for g in range(WORKERs):
                    for slot in range(1, SLOTs):
                        model.addConstr(model_slot[g, slot] - model_slot[g, slot - 1] <= 1 + MODELs - MODELs * transition_slot[g, slot])
                        model.addConstr(model_slot[g, slot] - model_slot[g, slot - 1] >= MODELs * transition_slot[g, slot] - MODELs - 1)
                        model.addConstr(model_slot[g, slot] - model_slot[g, slot - 1] <= MODELs * transition_slot[g, slot] - 1)
                        model.addConstr(model_slot[g, slot] - model_slot[g, slot - 1] >= 1 - MODELs * transition_slot[g, slot])

                for g in range(WORKERs):
                    for slot in range(SLOTs):
                        model.addConstr(completion_slot[g, slot] == quicksum(groups[i] * x[g, i, j] for i in range(N) for j in range(slot + 1)))

                for g in range(WORKERs):
                    for slot in range(SLOTs):
                        model.addConstr(penalty_slot[g, slot] == completion_slot[g, slot] + MODEL_SWAP_TIME * transition_slot[g, slot] - slo_slot[g, slot])

                for g in range(WORKERs):
                    for slot in range(SLOTs):
                        model.addConstr(penalty_slot[g, slot] <= 0)

                model.setObjective(quicksum(penalty_slot[g, slot] for g in range(WORKERs) for slot in range(SLOTs)), GRB.MINIMIZE)

                model.optimize()

                # [중요 변경 2] 성공 시에만 큐를 비우고 다시 채웁니다.
                if model.Status == GRB.OPTIMAL:
                    print("Optimal solution found for LP!")
                    
                    # 1. 기존 큐 비우기 (객체는 유지, 내용만 삭제)
                    for vq in vqs:
                        vq.groups.clear()
                    
                    # 2. 결과대로 다시 채우기
                    for g in range(WORKERs):
                        # 각 워커(GPU)별로 슬롯 순서대로 그룹을 찾아 넣어야 함
                        # 주의: 슬롯 순서(j)대로 정렬해서 넣어야 합니다.
                        
                        worker_groups = []
                        for slot in range(SLOTs):
                            for i in range(N):
                                var_name = f"x[{g},{i},{slot}]"
                                var = model.getVarByName(var_name)
                                if var and abs(var.X) > 0.5: # 0.5보다 크면 1로 간주
                                    worker_groups.append(group_idx_bimap.inv[i])
                        
                        # 찾은 순서대로 VQ에 추가
                        for grp in worker_groups:
                            vqs[g].groups.append(grp)
                            
                    return vqs
                
                else:
                    print(f"No optimal solution found (Status {model.Status}), reverting to EDF")
                    # 실패했으므로 vqs는 건드리지 않았음 (원본 유지)
                    return self._reorder_edf(vqs)

        except gp.GurobiError as e:
            print(f"Gurobi Error: {e}")
            return self._reorder_edf(vqs)
        except Exception as e:
            print(f"Unexpected Error in LP: {e}")
            return self._reorder_edf(vqs)