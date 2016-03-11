#!/usr/bin/env python
import sys, os, time, IPython
sys.path.append('../scripts')
from shapely.topology import *
import numpy as np
import matplotlib.pyplot as plt
from runPhase1 import *
from Planner import *
from simUtils import *
from model_fit import *
# from figures import SIZE, BLUE, GRAY
# from shapely.geometry import Point
# from descartes import PolygonPatch

# data1 = pickle.load(open("../scripts/saved_palpation_data/single_row_raster_100x.p", "rb"))
# # data2 = pickle.load(open("probe_data_L2R.p", "rb"))
# xdata,zdata=get_stiffness_data(data1)
# model=fit_measmodel(xdata,zdata)
# mod=plot_model(data1,model,scale=True)
# print mod
def run_single_phase2_simulation(phantomname, dirname, AcFunction=MaxVar_GP, control='Max', plot=False, exp=False):
    #if exp==True: 
    from expUtils import *
    bounds=calculate_boundary("../scripts/env_registration.p")
    #else:
    #    bounds=((-.08,.08),(-.08,.08))
    print(bounds)
    # grid resolution: should be same for plots, ergodic stuff
    gridres = 200

    # initialize workspace object
    workspace = Workspace(bounds,gridres)
    print workspace.bounds
    # plotSimulatedStiffnessMeas(phantomname, workspace, ypos=0, sensornoise = .05)

    # set level set to look for-- this should correspond to something, max FI?
    level = .7 #pick something between min/max deflection

    plot_data = None
    means = []
    sigmas = []
    acqvals = []
    healthyremoveds=[]
    tumorlefts=[]
    sampled_points = []
    measures = []

    directory=dirname
    if not os.path.exists(directory):
        os.makedirs(directory)

    ###############
    #Initializing
    ###############
    next_samples_points = randompoints(bounds, 10) #randompoints(bounds, 100)

    # collect initial meausrements
    if exp==True: 
        meas = getRecordedExperimentalStiffnessMeas(next_samples_points)
    else:
        
        meas1 = getSimulateStiffnessMeas(phantomname, next_samples_points)
    measnew=meas
    print 'expmeas=',meas
    print 'simmeas=',getSimulateStiffnessMeas(phantomname, next_samples_points)
    for j in range (10): #(1,100,1)
        # print "iteration = ", j
        # collect measurements

        # update the GP model    
        gpmodel = update_GP(meas)

        # use GP to predict mean, sigma on a grid
        mean, sigma = get_moments(gpmodel, workspace.x)

        # evaluate selected aqcuisition function over the grid
        if AcFunction==UCB_GPIS:
            acquisition_par=.05
        elif AcFunction==UCB_GP:
            acquisition_par=.7
        elif AcFunction==MaxVar_plus_gradient:
            acquisition_par=.6
        elif AcFunction==UCB_GPIS_implicitlevel:
            if exp==True:
                acquisition_par=[.1,.5]
            else:
                acquisition_par=[.6,.5]
                
        else:
            acquisition_par=0
        xgrid, AqcuisFunction = AcFunction(gpmodel, workspace, level=level, acquisition_par=acquisition_par)

        acqvals.append(AqcuisFunction)
        means.append(mean)
        sigmas.append(sigma)
        measures.append(meas)
        healthyremoved,tumorleft= evalerror(phantomname, workspace, mean,sigma,level)
        healthyremoveds.append(healthyremoved)
        tumorlefts.append(tumorleft)

        # select next sampling points. for now, just use Mac--dMax and Erg need work.
        bnd=getLevelSet (workspace, mean, level)
        # print 'max'  
        # if 
        # if len(bnd)==0:
        #     next_samples_points = randompoints(bounds, 1)

        if control=='Max':         
            # print 'max'   
            #     next_samples_points = maxAcquisition(workspace, AqcuisFunction,
            #                                             numpoints=1)
            # elif control=='batch'
            next_samples_points = batch_optimization(gpmodel, workspace, AcFunction, 10, level=level, acquisition_par=acquisition_par)
        # elif control=='dMax':
        #     next_samples_points = dmaxAcquisition(gpmodel, workspace, AcFunction, meas[-1][0:2],
        #                                         numpoints=10, level=level)
        else:
            next_samples_points=randompoints(bounds,1)
            print 'RANDOM'
        
        # Plot everything
        if plot==True:
            time.sleep(0.0001)
            plt.pause(0.0001)  
            plot_data = plot_beliefGPIS(phantomname, workspace, mean, sigma,
                                      AqcuisFunction, measnew,
                                      directory, [healthyremoveds,tumorlefts],plot_data,level=level,
                                      iternum=j, projection3D=False)

        if exp==True:
            measnew = getRecordedExperimentalStiffnessMeas(next_samples_points)#np.zeros(next_samples_points.shape)

        else:
            measnew = getSimulateStiffnessMeas(phantomname, next_samples_points)

        #   concatenate measurements to prior measurements
        meas = np.append(meas,measnew,axis=0)

    # plt.show(block=False)
    return means, sigmas, acqvals, measures, healthyremoveds, tumorlefts, j

def evalerror(tumor,workspace,mean,variance,level):
    # see: http://toblerity.org/shapely/manual.html
    boundaryestimate = getLevelSet (workspace, mean, level)
    boundaryestimateupper = getLevelSet (workspace, mean+variance, level)
    boundaryestimatelower = getLevelSet (workspace, mean-variance, level)
    #boundaryestimate=boundaryestimateupper
    GroundTruth = np.vstack((tumor,tumor[0]))
    GroundTruth=Polygon(GroundTruth)
    if len(boundaryestimate)>0:

        boundaryestimate=Polygon(boundaryestimate)
        try: 
            healthyremoved=boundaryestimate.difference(GroundTruth) # mislabeled data ()
            #boundaryestimate.difference(GroundTruth) #mislabeled as tumor--extra that would be removed
                
            tumorleft=GroundTruth.difference(boundaryestimate) # mislbaled as not-tumor--would be missed and should be cut out
            #correct=boundaryestimate.intersection(GroundTruth) #correctly labeled as tumor
            healthyremoved=healthyremoved.area
            tumorleft=tumorleft.area
        except TopologicalError:
            healthyremoved=.100
            tumorleft=.100

    else:
        healthyremoved=.100
        tumorleft=.100
    return healthyremoved,tumorleft

# def plotbounds(tumor,workspace,mean,sigma,level):
#     # see: http://toblerity.org/shapely/manual.html
#     GroundTruth = np.vstack((tumor,tumor[0]))
#     # GroundTruth=Polygon(GroundTruth)

#     boundaryestimateupper = getLevelSet (workspace, mean+sigma, level)
#     boundaryestimatelower = getLevelSet (workspace, mean-sigma, level)

#     boundaryestimate = getLevelSet (workspace, mean, level)
#     if boundaryestimate.shape[0]>0:
#         plt.plot(boundaryestimateupper.T[0], boundaryestimate.T[1], '-.',color='r',
#                      linewidth=1, solid_capstyle='round', zorder=2)
#         plt.plot(boundaryestimatelower.T[0], boundaryestimate.T[1], '-.',color='g',
#                      linewidth=1, solid_capstyle='round', zorder=2)
#         plt.plot(boundaryestimate.T[0], boundaryestimate.T[1], '-.',color='b',
#                      linewidth=1, solid_capstyle='round', zorder=2)
        
#     plt.plot(GroundTruth.T[0], GroundTruth.T[1], '-.',color='m',
#                  linewidth=1, solid_capstyle='round', zorder=2)


def save_data(arr, dirname, name):
    directory=dirname

    if not os.path.exists(directory):
        os.makedirs(directory)
    f = open(directory+'/data_'+name, 'w')
    f.write(str(np.array(arr).tolist()))
    f.close()
    return

NUM_EXPERIMENTS = 5


tumors = [squaretumor,rantumor]              # add another model ?
stops = [[6.4, 0.01, 3.6, 0.0],
         [0.0, 0.36, 0.0, 0.0]]                 # TODO change 0.0's (variance is not monotonically decreasing)
# textures = ["_lam", "_text", "_spec", "_st"]
    # lambert, texture, specular, specular + texture

# acquisition functions:  MaxVar_GP, UCB_GP, EI_GP, UCB_GPIS, EI_IS, MaxVar_plus_gradient
# MaxVar_plus_gradient(model, workspace, level=0, x=None, acquisition_par=0,numpoints=1)
aqfunctions = [UCB_GPIS_implicitlevel]#,MaxVar_plus_gradient,UCB_GP,UCB_GPIS,MaxVar_GP]
aqfunctionsnames = ["UCB_GPIS_implicitlevel"]#,"MaxVar_plus_gradient","UCB_GP", "UCB_GPIS", "MaxVar_GP"]#, "random"]

controls =["Max"]

def save_table(table, name):
    f = open(name + ".csv", 'wb')
    # header
    f.write(",,UCB_GPIS_implicitlevel,UCB_GBIS,MaxVar_plus_gradient,UCB_GB,MaxVar_GP\n")
    # data in table
    # f.write("flat,lam,{},{}\n".format(table[0][0], table[0][1]))
    f.write(",tumor1: iterations,{},{},{}\n".format(table[0][0], table[0][1],table[0][2],table[0][3],table[0][4]))
    f.write(",tumor1: healthy tissue removed,{},{},{}\n".format(table[1][0], table[1][1],table[1][2],table[1][3],table[1][4]))
    f.write(",tumor1: tumor left behind,{},{},{}\n".format(table[2][0], table[2][1],table[2][2],table[2][3],table[2][4]))

    f.write(",tumor2: iterations,{},{},{}\n".format(table[3][0], table[3][1],table[3][2],table[3][3],table[3][4]))
    f.write(",tumor2: healthy tissue removed,{},{},{}\n".format(table[4][0], table[4][1],table[4][2],table[4][3],table[4][4]))
    f.write(",tumor2: tumor left behind,{},{},{}\n".format(table[5][0], table[5][1],table[5][2],table[5][3],table[5][4]))

    # f.write(",st,{},{}\n".format(table[0][0], table[0][1]))

    # f.write("S,lam,{},{}\n".format(table[1][0], table[1][1]))
    # f.write(",text,{},{}\n".format(table[1][0], table[1][1]))
    # f.write(",spec,{},{}\n".format(table[1][0], table[1][1]))
    # f.write(",st,{},{}\n".format(table[1][0], table[1][1]))

    f.close()
    return


def plot_error(errorsrem,errorsleft,labels):    
    #for e in errors:
    #         plt.plot(e)
    fig = plt.figure(figsize=(3, 9))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax=[ax1,ax2,ax3]
    tum1left = errorsleft[0]
    tum2left = errorsleft[1]
    tum1rem = errorsrem[0]
    tum2rem = errorsrem[1]
    colors = ['blue','red','green','orange','magenta']

    # for i in range(0,tum1left.shape[0]):
    #     for j in range(0,tum1left.shape[1]):
    #         expl=tum1left[i][j]
    #         expl=np.mean(expl,axis=0)
    #         #for e in range(0,tum1left.shape[2]):
    #         #    exp = tum1left[i][j][e]
    #         #    print exp.shape
    #         ax[0].plot(expl,color=colors[i])
    for i in range(0,tum1rem.shape[0]):
        for j in range(0,tum1rem.shape[1]):
            expl=tum1left[i][j]
            expl=np.mean(expl,axis=0)

            expr=tum1rem[i][j]
            expr=np.mean(expr,axis=0)
            #or e in range(0,tum1rem.shape[2]):
            #    exp = tum1rem[i][j][e]
            ax[0].plot(expl,color=colors[i])
            ax[1].plot(expr,color=colors[i])
            ax[2].plot(expr+expl,color=colors[i])
    ym=.08*.08
    ym=.001
    ax[0].set_ylim(.00, ym)
    ax[0].set_title('error_leftover')
    ax[1].set_ylim(.00, ym)
    ax[1].set_title('error_removed')
    ax[2].set_ylim(.00, ym)
    ax[2].set_title('error_leftover+error_removed')
    plt.xlabel("Iterations")
    #ym=.08*.08
    #plt.ylim(.00, ym)

    # plt.xlim(0, 100)
    plt.ylabel(" Error")
    plt.legend(labels.flatten(), loc='upper right')
    # plt.savefig("image_pairs/"+surface_name+'/'+name)
    # plt.close()
    plt.show()
    return


def run_phase2_full():

    # iter_table = np.zeros((len(aqfunctions)+len(controls),len(tumors)))
    error_table = np.zeros((len(tumors)*3,len(aqfunctions)+len(controls)))
    # print iter_table.shape
    # error_table1 = np.zeros((2, len(aqfunctions)+len(controls)))
    tumorerrlistleft=[]
    tumorerrlistremoved=[]
    for i, tumor in enumerate(tumors):
        acqerrlistleft=[]
        acqerrlistremoved=[]
        aclabellist=[]
        for j, acq in enumerate(aqfunctions):
            conterrlistleft=[]
            conterrlistremoved=[]
            contlabellist=[]  
            for m, cont in enumerate(controls):
                # if i*len(textures) + j != 0:        # Use this to run only the nth surface
                #     continue
                its = [0.0, 0.0]
                # errors_per_method = []
                experrlistleft=[]
                experrlistremoved=[]
                for k in range(NUM_EXPERIMENTS): # repeat experiment 5 times
                    # disparityMeas = None
                    # for l, method in enumerate(methods):
                    start = time.time()
                    dirname = str(i) + '_' + aqfunctionsnames[j] + '_' + cont
                    means, sigmas, acqvals, measures, healthyremoved, tumorleft, num_iters = run_single_phase2_simulation(tumor, dirname, AcFunction=acq, control=cont, plot=True, exp=True)
                    plt.close() 
                    end = time.time()
                    time_elapsed = end - start # in seconds
                    # plot or save/record everything
                    # its[k] += num_iters / 5.0
                    save_data([means, sigmas, acqvals, measures, healthyremoved, tumorleft, num_iters], dirname, str(k))
                    # iter_table[j + m][i]+= num_iters / float(NUM_EXPERIMENTS)
                    # 3=num of errors in table
                    error_table[i+i*2+0][j+m]+= num_iters / float(NUM_EXPERIMENTS)
                    error_table[i+i*2+1][j+ m]+= healthyremoved[-1] / float(NUM_EXPERIMENTS)
                    error_table[i+i*2+2][j+ m]+= tumorleft[-1] / float(NUM_EXPERIMENTS)
                    experrlistleft.append(tumorleft)  
                    experrlistremoved.append(healthyremoved)  
                    print k

                print error_table
                # save_table(iter_table, "phase2_iterations")
                # save_table(error_table, "phase2_errors")
                # save_table(error_table1, "phase2_errors1")
                conterrlistleft.append(experrlistleft)
                conterrlistremoved.append(experrlistremoved)
                contlabellist.append(dirname)
            acqerrlistleft.append(conterrlistleft)
            acqerrlistremoved.append(conterrlistremoved)
            aclabellist.append(contlabellist)
        # plot_error(errors_per_method, "phase1_error_exp"+str(k), surf+text)

        tumorerrlistleft.append(acqerrlistleft)
        tumorerrlistremoved.append(acqerrlistremoved)
    return np.array(tumorerrlistleft),np.array(tumorerrlistremoved),np.array(aclabellist)



if __name__ == "__main__":
    outleft,outrem,aclabellist=run_phase2_full()
    plot_error(outrem,outleft,aclabellist)

