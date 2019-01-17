### ensemble을 통해 얻은 영역들 중에 최종 영역을 선택하기 위해, 영역들간 중복(겹치는 구간)제거 기준 정의 ###
MERGE_INTERSECT_AREA_RATE = 75 #두 영역간 중복(겹치는 구간)제거(병합) 기준율
NO_MERGE_INTERSECT_AREA_RATE = 90 #두 영역간 선택시 병합 여부 기준율(해당 기준 이상 시 병합없이 하나를 선택)
                                  # 병합제거시 Score는 두 영역 score의 평균으로 한다.
    

########## 2D list에서 특정 값이 위치한 index return ####################
def find(searchList, elem):
    for ix, row in enumerate(searchList):
        for iy, i in enumerate(row):
            if i==elem:
                #print('{},{}'.format(ix,iy))
                return [ix,iy]
    return []    

######2개 영역의 겹치는 면적을 구한다. ############
def area(a, b):  # returns 0 if rectangles don't intersect
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    if (dx>=0) and (dy>=0):
        return dx*dy
    else:
        return 0

######2개 영역의 겹치는 면적에 대해 각 영역에 대한 비율을 구한다. ############
def area_rate(intersect_area, a, b):  # returns [] if rectangles don't intersect
    if intersect_area > 0:
        a_rate = float('{:.2f}'.format((intersect_area/((a[2]-a[0]) * (a[3]-a[1]))) * 100))
        b_rate = float('{:.2f}'.format((intersect_area/((b[2]-b[0]) * (b[3]-b[1]))) * 100))
        return [a_rate, b_rate]
    else:
        return []

    
    ############ensemble 모델들에서 검출된 영역들에 대해서 중복영역 등을 고려하여 최종 영역을 선택 한다. ###############
# 선택기준                                                                                                    # 
# 1. 영역들 간 중복영역의 비율이 제거(또는 병합)대상 기준(MERGE_INTERSECT_AREA_RATE, 현75%)이 되는 영역들 중에서   #
# 2. 비 병합(병합없이 제거)기준(NO_MERGE_INTERSECT_AREA_RATE)에 따라 영역을 병합, 제거하여 선택                  #
# 3. select_deteced_list는 병합을 하지 않고 두 영역 중 제거 선택만...                                           #  
#    select_deteced_list_wih_merge는 기준에 따라 두 영역의 병합 선택을 적용                                     #
##############################################################################################################
####영역 리스트에 해당하는 선택된 index 목록을 return 한다. ####
def select_deteced_list(d_list, score_list):
    sel_idx_status = [0]*len(d_list)
    return sel_idx_status

    max_intersect_area = 0
    compare_idx = 0

    #print(sel_idx_status)

    for idx_i in range(len(d_list)):
        if sel_idx_status[idx_i] == 0:
            max_intersect_area = 0
            compare_idx = 0
            for idx_j in range(len(d_list)):
                if idx_i != idx_j and sel_idx_status[idx_j] == 0:
                    intersect_area = area(d_list[idx_i], d_list[idx_j])
                    if max_intersect_area < intersect_area:
                        max_intersect_area = intersect_area
                        compare_idx = idx_j

            area_rate_list = area_rate(max_intersect_area, d_list[idx_i], d_list[compare_idx])

            if len(area_rate_list) > 0:
                #print('select intersect rate : {}'.format(area_rate_list[0]))
                #print('compare intersect rate : {}'.format(area_rate_list[1]))

                if area_rate_list[0] >= MERGE_INTERSECT_AREA_RATE and area_rate_list[1] >= MERGE_INTERSECT_AREA_RATE:
                    #겹치는 영역이 양쪽 모두 90% 이상일 경우
                    if area_rate_list[0] >= NO_MERGE_INTERSECT_AREA_RATE and area_rate_list[1] >= NO_MERGE_INTERSECT_AREA_RATE:
                        #score가 큰쪽을 선택
                        if score_list[idx_i] >= score_list[compare_idx]:
                            sel_idx_status[compare_idx] = -1 #score 작은쪽 제거
                        else:
                            sel_idx_status[idx_i] = -1 #score 작은쪽 제거
                    else: #겹치는 영역이 양쪽 모두 90% 이하일 경우
                        #겹치는 영역이 각 영역별 차지하는 비율로 볼때 작은 영역을 선택
                        if area_rate_list[0] > area_rate_list[1]:
                            sel_idx_status[idx_i] = -1
                        else:
                            sel_idx_status[compare_idx] = -1
                elif area_rate_list[0] >= MERGE_INTERSECT_AREA_RATE and area_rate_list[1] < MERGE_INTERSECT_AREA_RATE:
                        sel_idx_status[idx_i] = -1
                elif area_rate_list[0] < MERGE_INTERSECT_AREA_RATE and area_rate_list[1] >= MERGE_INTERSECT_AREA_RATE:
                        sel_idx_status[compare_idx] = -1
                        
    return sel_idx_status


####영역 리스트에 해당하는 선택된 index 목록을 return 한다. ####
def select_deteced_list_wih_merge(d_list, score_list):
    sel_idx_status = [0]*len(d_list)
    
    #return sel_idx_status
    
    max_intersect_area = 0
    compare_idx = 0

    #print(sel_idx_status)

    for idx_i in range(len(d_list)):
        if sel_idx_status[idx_i] == 0:
            max_intersect_area = 0
            compare_idx = 0
            for idx_j in range(len(d_list)):
                if idx_i != idx_j and sel_idx_status[idx_j] == 0:
                    intersect_area = area(d_list[idx_i], d_list[idx_j])
                    if max_intersect_area < intersect_area:
                        max_intersect_area = intersect_area
                        compare_idx = idx_j

            area_rate_list = area_rate(max_intersect_area, d_list[idx_i], d_list[compare_idx])

            if len(area_rate_list) > 0:
                #print('select intersect rate : {}'.format(area_rate_list[0]))
                #print('compare intersect rate : {}'.format(area_rate_list[1]))

                if area_rate_list[0] >= MERGE_INTERSECT_AREA_RATE and area_rate_list[1] >= MERGE_INTERSECT_AREA_RATE:
                    #겹치는 영역이 양쪽 모두 90% 이상일 경우
                    if area_rate_list[0] >= NO_MERGE_INTERSECT_AREA_RATE and area_rate_list[1] >= NO_MERGE_INTERSECT_AREA_RATE:
                        #score가 큰쪽을 선택
                        if score_list[idx_i] >= score_list[compare_idx]:
                            sel_idx_status[compare_idx] = -1 #score 작은쪽 제거
                        else:
                            sel_idx_status[idx_i] = -1 #score 작은쪽 제거
                            
                    elif area_rate_list[0] >= NO_MERGE_INTERSECT_AREA_RATE and area_rate_list[1] < NO_MERGE_INTERSECT_AREA_RATE:
                        sel_idx_status[idx_i] = -1
                    elif area_rate_list[0] < NO_MERGE_INTERSECT_AREA_RATE and area_rate_list[1] >= NO_MERGE_INTERSECT_AREA_RATE:        
                        sel_idx_status[compare_idx] = -1
                    else: #겹치는 영역이 양쪽 모두 90% 이하일 경우
                        #두 영역을 합친다.
                        merged_area = merge_area(d_list[idx_i], d_list[compare_idx])
                        #겹치는 영역이 각 영역별 차지하는 비율로 볼때 작은 영역을 선택
                        if area_rate_list[0] > area_rate_list[1]:
                            sel_idx_status[idx_i] = -1
                            d_list[compare_idx] = merged_area.copy() #합쳐진 영역으로 업데이트
                            #Score는 합쳐진 두 영역 score의 평균으로 업데이트
                            score_list[compare_idx] = float('{:.3f}'.format((score_list[idx_i]+score_list[compare_idx])/2))
                        else:
                            sel_idx_status[compare_idx] = -1
                            d_list[idx_i] = merged_area.copy() #합쳐진 영역으로 업데이트
                            #Score는 합쳐진 두 영역 score의 평균으로 업데이트
                            score_list[idx_i] = float('{:.3f}'.format((score_list[idx_i]+score_list[compare_idx])/2))
                elif area_rate_list[0] >= MERGE_INTERSECT_AREA_RATE and area_rate_list[1] < MERGE_INTERSECT_AREA_RATE:
                    sel_idx_status[idx_i] = -1
                    if area_rate_list[0] < NO_MERGE_INTERSECT_AREA_RATE:
                        #두 영역을 합친다.
                        merged_area = merge_area(d_list[idx_i], d_list[compare_idx])
                        d_list[compare_idx] = merged_area.copy() #합쳐진 영역으로 업데이트
                        #Score는 합쳐진 두 영역 score의 평균으로 업데이트
                        score_list[compare_idx] = float('{:.3f}'.format((score_list[idx_i]+score_list[compare_idx])/2))
                elif area_rate_list[0] < MERGE_INTERSECT_AREA_RATE and area_rate_list[1] >= MERGE_INTERSECT_AREA_RATE:
                    sel_idx_status[compare_idx] = -1
                    if area_rate_list[1] < NO_MERGE_INTERSECT_AREA_RATE:
                        #두 영역을 합친다.
                        merged_area = merge_area(d_list[idx_i], d_list[compare_idx])
                        d_list[idx_i] = merged_area.copy() #합쳐진 영역으로 업데이트
                        #Score는 합쳐진 두 영역 score의 평균으로 업데이트
                        score_list[idx_i] = float('{:.3f}'.format((score_list[idx_i]+score_list[compare_idx])/2))                        
                        
    return sel_idx_status

#겹치는 영역이 있는 2개의 영역을 합침(일정 기준이 넘어설 경우), 쓰이지는 않음
def merge_area(area1, area2):
    new_area = area1.copy()
    if area1[0] > area2[0]:
        new_area[0] = area2[0]
        
    if area1[1] > area2[1]:
        new_area[1] = area2[1]
        
    if area1[2] < area2[2]:
        new_area[2] = area2[2]
        
    if area1[3] < area2[3]:
        new_area[3] = area2[3]
    
    return new_area