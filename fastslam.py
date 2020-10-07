# -*- coding: utf-8 -*-
from read_data import read_world, read_sensor_data
from misc_tools import *
import numpy as np
import math
import copy
#Настройки отображения
plt.axis([-1, 12, 0, 10])
plt.ion()
plt.show()

def initialize_particles(num_particles, num_landmarks):
    #Инициализация частиц [0,0,0] с пустой картой

    particles = []

    for i in range(num_particles):
        particle = dict()

        #Начальная позиция робота[0,0,0]
        particle['x'] = 0
        particle['y'] = 0
        particle['theta'] = 0

        #Начальный вес
        particle['weight'] = 1.0 / num_particles
        
        #История частицы (Все позиции)
        particle['history'] = []

        #Начальные ориентиры частицы
        landmarks = dict()

        for i in range(num_landmarks):
            landmark = dict()

            #Положение и разброс для ориентиров
            landmark['mu'] = [0,0]
            landmark['sigma'] = np.zeros([2,2])
            landmark['observed'] = False

            landmarks[i+1] = landmark

        #Добавляем ориентиры к частице
        particle['landmarks'] = landmarks

        #Добавляем частицу в массив
        particles.append(particle)

    return particles

def motion_model_sample(Odo_reading, particles):
    # Обновление положений частиц на основе одометрии
    # прошлого положения и шума 

    d_rot1 = Odo_reading['r1']
    d_trans = Odo_reading['t']
    d_rot2 = Odo_reading['r2']

    # Параметры шума движения: [alpha1, alpha2, alpha3]
    noise = [0.1, 0.05, 0.05]

    '''реализовать функцию'''
    '''***             ***'''
    new_particles = []
    for particle in particles:
        particle['history'].append([particle['x'], particle['y']])
        particle['theta'] = particle['theta']+np.random.normal(d_rot1,noise[1])
        particle['x'] = particle['x']+ np.random.normal(d_trans,noise[2])*np.cos(particle['theta'])
        particle['y'] = particle['y']+np.random.normal(d_trans,noise[2])*np.sin(particle['theta'])
        particle['theta'] = particle['theta']+np.random.normal(0,noise[0])+np.random.normal(d_rot2,noise[1])
        new_particles.append(particle)
    return new_particles


def measurement_model_partland(particle, landmark):
    #Вспомогательная функция для вычисления расстояния от частицы до ориентира
    #а также якобиана этого измерения

    px = particle['x']
    py = particle['y']
    ptheta = particle['theta']

    lx = landmark['mu'][0]
    ly = landmark['mu'][1]

    #Вычисление расстояния и направления
    meas_range_exp = np.sqrt( (lx - px)**2 + (ly - py)**2 )
    meas_bearing_exp = math.atan2(ly - py, lx - px) - ptheta

    h = np.array([meas_range_exp, meas_bearing_exp])

    # Вычисление якобиана функции измерения h

    
    H = np.zeros((2,2))
    H[0,0] = (lx - px) / h[0]
    H[0,1] = (ly - py) / h[0]
    H[1,0] = (py - ly) / (h[0]**2)
    H[1,1] = (lx - px) / (h[0]**2)

    return h, H

def eval_sensor_model_particle(part_set, sens_data):
    #Корректировка позиций ориентиров
    #вычисление весов частиц

    #шум датчика измерения
    Qt = np.array([[0.1, 0],\
                    [0, 0.1]])

    #Измеренные параметры ориентира id, расстояние, направление
    l_ids = sens_data['id']
    l_ranges = sens_data['range']
    l_bearings = sens_data['bearing']

    #Обновляем ориентиры и вычисляем веса частиц
    for singlepart in part_set:

        landmark_set = singlepart['landmarks']

        px = singlepart['x']
        py = singlepart['y']
        ptheta = singlepart['theta'] 

        #Цикл по всем увиденным в измерении ориентирам
        for i in range(len(l_ids)):

            #текущий ориентир
            lm_id = l_ids[i]
            single_landmark = landmark_set[lm_id]
            
            #измеренное растояние и угол до ориентира
            measured_range = l_ranges[i]
            measured_bearing = l_bearings[i]

            if not single_landmark['observed']:
                # если видим ориентир в первый раз
                
                # оцениваем позицию ориентира и вычисляем матрицу ковариации. 
                # Можно использовать вспомагательную функцию 'meas_model' выше
                '''реализовать функцию'''
                '''***             ***'''
                lx = px + measured_range * np.cos(ptheta + measured_bearing)
                ly = py + measured_range * np.sin(ptheta + measured_bearing)
                single_landmark['mu'] = [lx, ly]
            
                h, H = measurement_model_partland(singlepart, single_landmark)
                H_obr= np.linalg.inv(H)
                single_landmark['sigma']=np.dot(np.dot(Qt, H_obr),np.transpose(H_obr))
                
                single_landmark['observed'] = True

            else:
                # если ориентир уже есть в базе данных

                # обновляем позицию ориентира и его ковариацию
                # Можно использовать вспомагательную функцию 'meas_model' выше
                # вычисляем вес частицы: particle['weight'] = ...
                '''реализовать функцию'''
                '''***             ***'''
                h, H = measurement_model_partland(singlepart, single_landmark)

                l_s=single_landmark['sigma']
                S=np.dot(np.dot(H,l_s),np.transpose(H))+Qt
                K=np.dot(np.dot(l_s,np.transpose(H)),np.linalg.inv(S))
                
                diff = np.array([measured_range - h[0], angle_diff(measured_bearing,h[1])])
                single_landmark['mu']=single_landmark['mu']+np.dot(K,diff)
                single_landmark['sigma']= np.dot(np.eye(2) - np.dot(K,H),l_s)

                #вычисление веса
                valid = 1 / np.sqrt(np.linalg.det(2* math.pi * Qt))
                expo = -0.5 * np.dot(np.dot(diff, np.linalg.inv(Qt)),np.transpose(diff))
                singlepart['weight'] = singlepart['weight'] * valid * np.exp(expo)
                

    #нормализуем веса
    normalizer = sum([singlepart['weight'] for singlepart in part_set])

    for singlepart in part_set:
        singlepart['weight'] = singlepart['weight'] / normalizer
    return part_set

def particles_resampling(particle_set):
    # Возвращаем новый набор частиц после ресемплинга


    new_particle_set = []

    '''реализовать функцию'''
    '''***             ***'''

    # для переноса частицы в новый набор использовать следующую функцию:
    # new_singlepart = copy.deepcopy(particle_set[i])
    # ...
    # new_particle_set.append(new_singlepart)

    mx= 2*best_particle(particle_set)['weight']
    betta = np.random.uniform(0,mx)
    i = 0

    for particle in particle_set:

        while betta > particle_set[i]['weight']:
            i = (i + 1)%(len(particle_set))
            betta = betta - particle_set[i]['weight']

        new_particle = copy.deepcopy(particle_set[i])
        new_particle_set.append(new_particle)

        betta  = betta  + np.random.uniform(0,mx)

    return new_particle_set

def main():

    print ("Reading landmark positions")
    landmarks = read_world("../data/world.dat")

    print ("Reading sensor data")
    sensor_readings = read_sensor_data("../data/sensor_data.dat")

    num_particles = 100
    num_landmarks = len(landmarks)

    #создаем начальный набор частиц
    particles = initialize_particles(num_particles, num_landmarks)

    #запускаем  FastSLAM
    for timestep in range(len(sensor_readings)//2):

        #перемещение частиц с моделью движения
        motion_model_sample(sensor_readings[timestep,'odometry'], particles)

        #использууем измерение для обновления ориентиров и вычисления весов
        eval_sensor_model_particle(particles, sensor_readings[timestep, 'sensor'])

        #отрисовка текущего состояния
        plot_state(particles, landmarks)

        #вычисляем новый набор частиц после ресемплинга
        particles = particles_resampling(particles)

    plt.show('hold')

if __name__ == "__main__":
    main()
