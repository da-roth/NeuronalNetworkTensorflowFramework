#Packages
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from matplotlib import pyplot as plt 
import tensorflow_probability as tfp
import time
from tensorflow.python.ops import init_ops
from tensorflow.python.training.moving_averages import assign_moving_average


def train_and_test_Multilevel(Generator, Regressor, TrainSettings):

    def train_and_test_MultilevelIntern(Regressor, TrainSettings, xi_list, phi_list, xi_approx, u_reference_GBM, u_reference_list, neurons, dtype=tf.float32):
    
        def _approximate_errors():
            gs_lr = sess.run([global_step[i] for i in range(amountNetworks)] + [learning_rate[i] for i in range(amountNetworks)])
            gs = gs_lr[:amountNetworks]
            lr = gs_lr[amountNetworks:]
            li_err = [0. for _ in range(amountNetworks)]
            li_err_kombination = 0.
            for _ in range(TrainSettings.mcRounds):
                li = sess.run([err_l_inf[i] for i in range(amountNetworks)], feed_dict={is_training: False})
                appr_ref_kombination = sess.run([approx[i] for i in range(amountNetworks)] + [reference, err_l_kombination], feed_dict={is_training: False})
                appr = appr_ref_kombination[:amountNetworks]
                ref = appr_ref_kombination[amountNetworks]
                li_kombination = appr_ref_kombination[-1]
                for i in range(amountNetworks):
                    li_err[i] = np.maximum(li_err[i], li[i])
                li_err_kombination = np.maximum(li_err_kombination, li_kombination)
            t_mc = time.time()
            file_out.write(f'{gs[0]}, {li_err_kombination}, {lr[0]}, {t1_train - t0_train}, {t_mc - t1_train}\n')
            file_out.flush()
        
        t0_train = time.time()
        is_training = tf.placeholder(tf.bool, [])

        amountNetworks = len(phi_list)
        u_approx = []
        for i in range(amountNetworks):
            u_approx.append(Regressor.neural_net(xi_list[i], xi_approx, neurons, is_training, f'u_approx_{i}', str(i), dtype=dtype))

        loss = [tf.reduce_mean(tf.squared_difference(u_approx[i], phi_list[i])) for i in range(amountNetworks)]

        approx = [tf.reduce_mean(u_approx[i]) for i in range(amountNetworks)]
        reference = tf.reduce_mean(u_reference_GBM)

        err = [tf.abs(u_approx[i] - u_reference_list[i]) for i in range(amountNetworks)]

        err_kombination = tf.abs(sum(u_approx) - u_reference_GBM)

        err_l_inf = [tf.reduce_max(err[i]) for i in range(len(err))]
        err_l_kombination = tf.reduce_max(err_kombination)

        lr = [TrainSettings.learningRateSchedule[0] for _ in range(amountNetworks)]
        step_rate = [TrainSettings.learningRateSchedule[2] for _ in range(amountNetworks)]
        decay = [TrainSettings.learningRateSchedule[1] for _ in range(amountNetworks)]
        global_step = [tf.Variable(1, trainable=False) for _ in range(amountNetworks)]
        increment_global_step = [tf.assign(global_step[i], global_step[i] + 1) for i in range(amountNetworks)]
        learning_rate = [tf.train.exponential_decay(lr[i], global_step[i], step_rate[i], decay[i], staircase=True) for i in range(amountNetworks)]
        optimizer = [tf.train.AdamOptimizer(learning_rate[i]) for i in range(amountNetworks)]
        update_ops = [tf.get_collection(tf.GraphKeys.UPDATE_OPS, f'u_approx_{i}') for i in range(amountNetworks)]
        
        train_op = []
        for i in range(len(update_ops)):
            with tf.control_dependencies(update_ops[i]):
                train_op.append(optimizer[i].minimize(loss[i], global_step[i])) 
                
        file_out = open(TrainSettings.FileName, 'w')
        file_out.write('step,li_err, learning_rate, time_train, time_mc  \n ')
            
        with tf.Session() as sess:
            if isinstance(TrainSettings.TrainingSteps, list) and len(TrainSettings.TrainingSteps) > 1:
                sess.run(tf.global_variables_initializer())
                for step in range(1,max(TrainSettings.TrainingSteps)):
                    if step % TrainSettings.testFrequency == 0:
                        print(step)
                        t1_train = time.time()
                        _approximate_errors()
                        t0_train = time.time()     
                    StillTrained = [x > step for x in TrainSettings.TrainingSteps]
                    train_op_updated = [train_op[i] for i in range(len(StillTrained)) if StillTrained[i]]
                    sess.run(train_op_updated, feed_dict={is_training:True})
            else:
                sess.run(tf.global_variables_initializer())
                for step in range(1,TrainSettings.TrainingSteps):
                    if step % TrainSettings.testFrequency == 0:
                        print(step)
                        t1_train = time.time()
                        _approximate_errors()
                        t0_train = time.time()    
                    sess.run(train_op, feed_dict={is_training:True})
            t1_train = time.time()
            _approximate_errors()
        file_out.close()

    # Settings import for training
    Generator.set_batch_sizes(TrainSettings.SamplesPerStep)
    Generator.set_dtype(tf.float32)

    #Model and training parameter specification  
    for i in range(1,2):
        #print(i)
        tf.reset_default_graph()
        tf.random.set_random_seed(i)
        with tf.Session()  as sess:
            dtype = tf.float32
            #Set network and training parameter (same number of training steps for each network)
            batch_sizes = TrainSettings.SamplesPerStep # original [75000, 1817, 690, 264, 93, 33, 12, 5]
            batch_size_approx= TrainSettings.nTest# original 2000000
            d = 5
            # Level adaptation parameter: steps = M ^ l
            N = Generator.StepsInitialLevel
            M = 2
            maximumLevel = len(batch_sizes) - 1 # P_0 + P_1-P_0, here 1 is the maximumLevel
            stepsPerLevel = [(M**i)*N for i in range(maximumLevel)]
            Generator.set_stepsPerLevel(stepsPerLevel )

            neurons = [Regressor.HiddenNeurons for _ in range(Regressor.HiddenLayers)] + [1]
            train_steps = TrainSettings.TrainingSteps # original 150000
            
            Ksteps_p1_p0=train_steps
            Ksteps_p0=   train_steps    
            mc_rounds, mc_freq = TrainSettings.mcRounds, TrainSettings.testFrequency # original  100, 10

            mc_samples_ref, mc_rounds_ref_p0, mc_rounds_ref_p1_p0 = 1, 1000000,1000000
            Generator.set_mc_samples_ref(mc_samples_ref)

            # Define the intervals for the parameters
            s_0_l = 80.0
            s_0_r = 120.0
            sigma_l = 0.1
            sigma_r = 0.2
            mu_l = 0.02
            mu_r = 0.05
            T_l = 0.9
            T_r = 1.0
            K_l = 109.0
            K_r = 110.0

            # Define the modifiers for the approximation intervals
            s_0_modifier = 0.4
            sigma_modifier = 0.01
            mu_modifier = 0.01
            T_modifier = 0.01
            K_modifier = 0.1

            # Use the modifiers to define the intervals for the approximations of the parameters
            s_0_l_approx = s_0_l + s_0_modifier
            s_0_r_approx = s_0_r - s_0_modifier
            sigma_l_approx = sigma_l + sigma_modifier
            sigma_r_approx = sigma_r - sigma_modifier
            mu_l_approx = mu_l + mu_modifier
            mu_r_approx = mu_r - mu_modifier
            T_l_approx = T_l + T_modifier
            T_r_approx = T_r - T_modifier
            K_l_approx = K_l + K_modifier
            K_r_approx = K_r - K_modifier

            # Training intervals
            s0 = tf.random_uniform((batch_sizes[0],1), minval=s_0_l,
                                            maxval=s_0_r, dtype=dtype)
            sigma=tf.random_uniform((batch_sizes[0],1),
                                        minval=sigma_l,maxval=sigma_r, dtype=dtype)
            mu=tf.random_uniform((batch_sizes[0],1),
                                        minval=mu_l,maxval=mu_r, dtype=dtype)
            T=tf.random_uniform((batch_sizes[0],1),
                                        minval=T_l,maxval=T_r, dtype=dtype)
            K=tf.random_uniform((batch_sizes[0],1),
                                minval=K_l,maxval=K_r, dtype=dtype)
            
            xi_level0=tf.reshape(tf.stack([s0,sigma,mu,T,K], axis=2), (batch_sizes[0],d))

            xi_list = []
            xi_list.append(xi_level0)

            loop_var_mc = []
            loop_var_mc.append((tf.constant(0),tf.ones((mc_samples_ref,batch_sizes[0], 1), dtype) * s0, tf.ones((mc_samples_ref,batch_sizes[0], 1), dtype) * sigma,tf.ones((mc_samples_ref,batch_sizes[0], 1), dtype) * mu,tf.ones((mc_samples_ref,batch_sizes[0], 1), dtype) * T,tf.ones((mc_samples_ref,batch_sizes[0], 1), dtype) * K))
            
            Generator.set_loop_var_mc(loop_var_mc)

            for i in range(1,len(batch_sizes)):
                s0_level_estimator = tf.stack((tf.random_uniform((batch_sizes[i],1), minval=s_0_l, maxval=s_0_r, dtype=dtype)))
                sigma_level_estimator = tf.random_uniform((batch_sizes[i],1), minval=sigma_l, maxval=sigma_r, dtype=dtype)
                mu_level_estimator = tf.random_uniform((batch_sizes[i],1), minval=mu_l, maxval=mu_r, dtype=dtype)
                T_level_estimator = tf.random_uniform((batch_sizes[i],1), minval=T_l, maxval=T_r, dtype=dtype)
                K_level_estimator = tf.random_uniform((batch_sizes[i],1), minval=K_l, maxval=K_r, dtype=dtype)
                xi_level_estimator= tf.reshape(tf.stack([s0_level_estimator, sigma_level_estimator, mu_level_estimator, T_level_estimator, K_level_estimator], axis=2), (batch_sizes[i], d))
                xi_list.append(xi_level_estimator)
                loop_var_mc.append((tf.constant(0),tf.ones((mc_samples_ref,batch_sizes[i], 1), dtype) * s0_level_estimator,tf.ones((mc_samples_ref,batch_sizes[i], 1), dtype) * s0_level_estimator, tf.ones((mc_samples_ref,batch_sizes[i], 1), dtype) * sigma_level_estimator,tf.ones((mc_samples_ref,batch_sizes[i], 1), dtype) * mu_level_estimator,tf.ones((mc_samples_ref,batch_sizes[i], 1), dtype) * T_level_estimator,tf.ones((mc_samples_ref,batch_sizes[i], 1), dtype) * K_level_estimator))

            # Approximation intervals
            s0_approx = tf.random_uniform((batch_size_approx,  1 ), 
                                        minval=s_0_l_approx,maxval=s_0_r_approx, dtype=dtype)
            sigma_approx=tf.random_uniform((batch_size_approx,  1 ), 
                                        minval=sigma_l_approx,maxval=sigma_r_approx, dtype=dtype)
            mu_approx=tf.random_uniform((batch_size_approx,1),
                                        minval=mu_l_approx,maxval=mu_r_approx, dtype=dtype)
            T_approx=tf.random_uniform((batch_size_approx,1),
                                        minval=T_l_approx,maxval=T_r_approx, dtype=dtype)
            K_approx=tf.random_uniform((batch_size_approx,1),
                                        minval=K_l_approx,maxval=K_r_approx, dtype=dtype)
            xi_approx=tf.reshape(tf.stack([s0_approx,sigma_approx,mu_approx,T_approx,K_approx], axis=2), (batch_size_approx,d))

            # References: Black-Scholes formula as reference
            tfd = tfp.distributions
            dist = tfd.Normal(loc=tf.cast(0.,tf.float32), scale=tf.cast(1.,tf.float32))
            d1=tf.math.divide(
            (tf.log(tf.math.divide(s0_approx,K_approx))+(mu_approx + 0.5*sigma_approx**2)*T_approx) , (sigma_approx*tf.sqrt(T_approx)))
            d2=tf.math.divide(
            (tf.log(tf.math.divide(s0_approx,K_approx))+(mu_approx - 0.5*sigma_approx**2)*T_approx) , (sigma_approx*tf.sqrt(T_approx)))

            u_reference= tf.multiply(s0_approx,(dist.cdf(d1)))-K_approx*tf.exp(-mu_approx*T_approx)*(dist.cdf(d2))

        u_list = []
        phi_list = []
        u_reference_list =[]

        u_list.append(tf.while_loop(lambda idx, p: idx < 1, Generator.MonteCarlo_loop_level0,(tf.constant(0), tf.zeros((batch_sizes[0], 1), dtype)))[1])
        phi_list.append(u_list[0] / tf.cast(1, tf.float32))
        u_reference_list.append(tf.multiply(s0_approx,(dist.cdf(d1)))-K_approx*tf.exp(-mu_approx*T_approx)*(dist.cdf(d2)))


        for i in range(1,len(batch_sizes)):
            u_list.append(tf.while_loop(lambda idx, p: idx < 1, lambda idx, p: Generator.MonteCarlo_loop_levelEstimator(idx, p, i), (tf.constant(0), tf.zeros((batch_sizes[i], 1), dtype)))[1])
            phi_list.append(u_list[i] / tf.cast(1, tf.float32))
            u_reference_list.append(xi_approx*0.)

        #Start training and testing                        
        train_and_test_MultilevelIntern(Regressor, TrainSettings,xi_list, phi_list, xi_approx, u_reference, u_reference_list, neurons, dtype)        