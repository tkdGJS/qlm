# 1. 이미지 파일 출력 설정 (PNG 파일로 저장)
#    - size: 이미지 크기 (가로 600, 세로 800)
#    - font: 글꼴 설정
set terminal pngcairo size 600,800 enhanced font 'Verdana,12'
set output 'execution_times.png'

# 2. 그래프 제목 및 라벨 설정
set title "Execution Time Distribution"
set ylabel "Execution Time (seconds)"

# 3. X축 설정 (1차원 분포이므로 X축 눈금 제거)
set xrange [0.5:1.5]  # X축 범위를 0.5 ~ 1.5로 좁게 고정
unset xtics           # [수정됨] X축 눈금 숫자(1, 2...)를 아예 없앰

# 4. 스타일 설정
set grid y            # Y축 방향으로만 격자(Grid) 표시
set style fill transparent solid 0.5 noborder  # 점을 반투명하게

# 5. 그리기 명령
#    - "execution_times_1.txt": 데이터 파일명 (환경에 맞게 변경 가능)
#    - using (1):1 : 모든 데이터를 X좌표 1인 지점에 일자로 찍음
plot "execution_times_1.txt" using (1):1 with points pt 7 ps 1.5 lc rgb "blue" title "Latency"