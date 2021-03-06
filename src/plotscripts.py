from utils import *
import numpy as np
from simUtils import *
from shapely.geometry import Point
from descartes import PolygonPatch

def plot_beliefGPIS(poly,workspace,mean,variance,aq,meas,dirname,errors,data=None, iternum=0, level=.4, tiltlev=0, projection3D=False):
    # parse locations, measurements, noise from data
    # gp data
    
    xx=workspace.xx
    yy=workspace.yy

    # print 'bnds=',(xx.min(), xx.max(), yy.min(),yy.max() )
    # GPIS = implicitsurface(mean,variance,level)
    lev=.5*(measmax-measmin)+measmin#(mean.max()-mean.min())+mean.min()

    if tiltlev>0:
        lev=.5*(mean.max()-mean.min())+mean.min()
        # offset=.006

    print lev

    boundaryestimate = getLevelSet (workspace, mean,lev, allpoly=True)

    boundaryestimateupper = getLevelSet (workspace, mean+variance,lev)#+1.96*np.sqrt(variance), level)
    #boundaryestimatelower = getLevelSet (workspace, mean-variance, level)
    mean = gridreshape(mean,workspace)

    variance = gridreshape(variance,workspace)
    aq = gridreshape(aq,workspace)
    # GPIS = gridreshape(GPIS,workspace)
    # print meas
    # for plotting, add first point to end
    # GroundTruth = np.vstack((poly,poly[0]))
    GroundTruth = poly
    grad = gradfd(mean,workspace)
    # grad=gridreshape(grad,workspace)
    if data is None:
        fig = plt.figure(figsize=(20, 4))
        if projection3D==True:
            ax1 = fig.add_subplot(161, projection='3d')
            ax2 = fig.add_subplot(162, projection='3d')
        else:
            ax1 = fig.add_subplot(161)
            ax2 = fig.add_subplot(162)
        ax3 = fig.add_subplot(163)
        ax4 = fig.add_subplot(164)
        ax5 = fig.add_subplot(165)
        ax6 = fig.add_subplot(166)

        fig.canvas.draw()
        plt.show(block=False)
        data = [fig, ax1, ax2, ax3, ax4, ax5,ax6]
    data[1].clear()
    data[2].clear()
    data[3].clear()
    data[4].clear()
    data[5].clear() 
    data[6].clear()
    # plot the mean
    if projection3D==True:
        data[1].plot_surface(xx, yy, mean, rstride=1, cstride=1,
                             cmap=cm.coolwarm, linewidth=0,
                             antialiased=False)

    else:
        data[1].imshow(np.flipud(mean), cmap=cm.coolwarm, vmax=measmax, vmin=measmin-.5,
                       extent=(xx.min(), xx.max(), yy.min(),yy.max() ))
        data[1].scatter(meas.T[0], meas.T[1], s=10,
                        cmap=cm.coolwarm)
        data[1].plot(meas.T[0], meas.T[1],'--k')
        #data[1].scatter(meas.T[0], meas.T[1], c=meas.T[2], s=20,
        #            cmap=cm.coolwarm)
    data[1].set_title("Stiffness Mean")
    data[1].set_xlabel('x')
    data[1].set_ylabel('y')
    data[1].set_ylim([yy.min(),yy.max()])
    data[1].set_xlim([xx.min(),xx.max()])                   
    data[4].set_aspect('equal')
    start, end = data[1].get_xlim()
    data[1].xaxis.set_ticks(np.arange(start, end, .01))
    #data[1].set_ylim([yy.min(),yy.max()])
    #data[1].set_xlim([xx.min(),xx.max()])
    # plot the uncertainty
    if projection3D==True:
        #lim=1
        cs1=data[2].plot_surface(xx, yy, variance, rstride=1, cstride=1,
                                 cmap=cm.Greys, linewidth=0,
                                 antialiased=False)
    else:
        data[2].imshow(np.flipud(variance), cmap=cm.Greys,
                       extent=(xx.min(), xx.max(), yy.min(),yy.max() ))
        data[2].scatter(meas.T[0], meas.T[1], s=10,
                        cmap=cm.coolwarm)
    
    data[2].set_title("Stiffness Variance")  
    data[2].set_xlabel('x')
    data[2].set_ylabel('y')
    data[2].set_ylim([yy.min(),yy.max()])
    data[2].set_xlim([xx.min(),xx.max()])
    start, end = data[2].get_xlim()
    data[2].xaxis.set_ticks(np.arange(start, end, .01))
    ################
    #plot acquiisition function
    ################
    data[3].set_title("Acquisition Fn.")
    cs=data[3].imshow(np.flipud(aq), cmap=cm.jet, extent=(xx.min(),
                                                            xx.max(),
                                                            yy.min(),yy.max())
    )
    data[3].plot(meas.T[0], meas.T[1])
    data[3].scatter(meas.T[0], meas.T[1],s=10,
                    cmap=cm.coolwarm)
    data[3].set_ylim([yy.min(),yy.max()])
    data[3].set_xlim([xx.min(),xx.max()])
    start, end = data[3].get_xlim()
    data[3].xaxis.set_ticks(np.arange(start, end, .01))
    data[4].set_title("Boundary and GT")

    ################
    #plot level sets/polygons
    ################
    # data[4].contour(xx, yy, mean, levels=[lev], color='m')
    offsetboundary=[]
    for i in range(0,boundaryestimate.shape[0]):
        bnd=boundaryestimate[i]
        if bnd.shape[0]>3:
            # data[4].plot(bnd[:,0], bnd[:,1], '-',color='k',
      #            linewidth=1, solid_capstyle='round', zorder=2)
            # for bn in bnd:
            bnd=Polygon(bnd)
            bnd=bnd.buffer(-offset)
            try:
                bnd=np.array(bnd.exterior.coords)
                data[4].plot(bnd.T[0], bnd.T[1], '-',color='r',
                    linewidth=1, solid_capstyle='round', zorder=2)
            except AttributeError:
                 bnd=[]
            offsetboundary.append(bnd)
    bndpoly=offsetboundary

    offsetboundaryupper=[]
    for i in range(boundaryestimateupper.shape[0]):
        bnd1=boundaryestimateupper[i]
        if bnd1.shape[0]>3:
            # data[4].plot(bnd1[:,0], bnd1[:,1], '-.',color='k',
            #            linewidth=1, solid_capstyle='round', zorder=2)
            bndupperpoly=Polygon(bnd1)
            bndupperpoly=bndupperpoly.buffer(-offset)
            try:
                bndupperpoly=np.array(bndupperpoly.exterior.coords)
                data[4].plot(bndupperpoly.T[0], bndupperpoly.T[1], '-.',color='r',
                         linewidth=1, solid_capstyle='round', zorder=2)
            except AttributeError:
                bndupperpoly=[]
            # offsetboundaryupper.append(bnd1)
    # bndpoly=offsetboundaryupper
    # if boundaryestimatelower.shape[0]>3:
    #     boundaryestimatelower=Polygon(boundaryestimatelower)
    #     bnd=boundaryestimatelower.buffer(-offset)
    #     try:
    #         boundaryestimatelower=np.array(bnd.exterior.coords)
    #         data[4].plot(boundaryestimatelower.T[0], boundaryestimatelower.T[1], '--',color='k',
    #             linewidth=1, solid_capstyle='round', zorder=2)
    #     except AttributeError:
    #         boundaryestimatelower=[]
        
    # data[4].plot(GroundTruth.T[0], GroundTruth.T[1], '-.',color='g',
    #              linewidth=1, solid_capstyle='round', zorder=2)

    data[4].legend( loc='upper right' )
    data[4].set_xlabel('x')
    data[4].set_ylabel('y')
    data[4].set_ylim([yy.min(),yy.max()])
    data[4].set_xlim([xx.min(),xx.max()])                   
    data[4].set_aspect('equal')
    start, end = data[4].get_xlim()
    data[4].xaxis.set_ticks(np.arange(start, end, .01))
    
    GT = Polygon(GroundTruth)
    patch1 = PolygonPatch(GT, alpha=0.2, zorder=1)
    data[4].add_patch(patch1)
    if len(bndpoly)>0:
        
        for b in bndpoly:
            if len(b)>3:
                bn=Polygon(b)
                patch2 = PolygonPatch(bn, fc='gray', ec='gray', alpha=0.2, zorder=1)
                data[4].add_patch(patch2)
                # if bn.is_ring:
                data[4].add_patch(patch2)
                c1 = GT.difference(bn)
                c2 = bn.difference(GT)

                if c1.geom_type == 'Polygon':
                    patchc = PolygonPatch(c1, fc='red', ec='red', alpha=0.5, zorder=2)
                    data[4].add_patch(patchc)
                elif c1.geom_type == 'MultiPolygon':
                    for p in c1:
                        patchp = PolygonPatch(p, fc='red', ec='red', alpha=0.5, zorder=2)
                        data[4].add_patch(patchp)
                if c2.geom_type == 'Polygon':
                    patchc = PolygonPatch(c2, fc='blue', ec='blue', alpha=0.5, zorder=2)
                    data[4].add_patch(patchc)
                elif c2.geom_type == 'MultiPolygon':
                    for p in c2:
                        patchp = PolygonPatch(p, fc='blue', ec='blue', alpha=0.5, zorder=2)
                        data[4].add_patch(patchp)
    data[5].plot(errors,color='red')
    # data[5].plot(errors[1],color='blue')

    data[5].set_xlabel("Iterations")
    data[5].set_ylabel(" Error")

    ym=.08*.08
    # plt.ylim(.00, .001)
    data[5].set_xlim(0, iternum)
    data[5].set_title("symmetric difference between boundary and GT")

    data[6].set_title("surface gradient")
    cs=data[6].imshow(np.flipud(grad), cmap=cm.jet, extent=(xx.min(),
                                                            xx.max(),
                                                            yy.min(),yy.max())
    )
    data[6].plot(meas.T[0], meas.T[1])
    data[6].scatter(meas.T[0], meas.T[1], s=10,
                    cmap=cm.coolwarm)
    data[6].set_ylim([yy.min(),yy.max()])
    data[6].set_xlim([xx.min(),xx.max()])
    start, end = data[6].get_xlim()
    data[6].xaxis.set_ticks(np.arange(start, end, .01))
    
    if (len(data) > 7):
        data[7].remove()
        data = data[:7]
    # cb2 = plt.colorbar(cs2, norm=norm)
    # cb2.set_label('${\\rm \mathbb{P}}\left[\widehat{G}(\mathbf{x}) = 0\\right]$')# # Define
    # data.append(cb2)
    
    # data[0].canvas.draw()

    data[0].savefig(dirname + '/' + str(iternum) + ".pdf", bbox_inches='tight')
    return data


    ########################## Plot Scripts
def plot_error(surface, workspace, mean, sigma, aq, meas, dirname=None, data=None,iternum=0, projection3D=True, plotmeas=True):
    # choose points to compare
    xx=workspace.xx
    yy=workspace.yy

    mean = gridreshape(mean,workspace)
    sigma = gridreshape(sigma,workspace)
    x = workspace.xlin
    y = workspace.ylin
    aq = gridreshape(aq,workspace)

    interpf = getInterpolatedStereoMeas(surface,workspace)
    disparity = getStereoDepthMap(surface)
    # disparity = gridreshape(disparity,workspace)
    #disparity = SimulateStereoMeas(surface, workspace)
    # xx, yy = np.meshgrid(x, y)

    interp = getInterpolatedGTSurface(surface, workspace)
    # interp=getInterpolatedObservationModel(surface)
    GroundTruth = getObservationModel(surface)
    # GroundTruth = gridreshape(GroundTruth,workspace)

    GroundTruth = interp(x,y)
    disparity = interpf(x,y)

    # GroundTruth=GroundTruth.reshape(xx.shape)
    # evaluate the Gaussian Process mean at the same points

    # evaluate the RMSerror
    error =np.sqrt((GroundTruth-np.squeeze(mean))**2)
    # GroundTruth = gridreshape(GroundTruth,workspace)
    # error = gridreshape(error,workspace)
    if data is None:
        fig = plt.figure(figsize=(24, 4))
        if projection3D==True:
            ax1 = fig.add_subplot(151, projection='3d')
            ax2 = fig.add_subplot(152, projection='3d')
            # ax3 = fig.add_subplot(153, projection='3d')
            # ax4 = fig.add_subplot(154, projection='3d')
            # ax5 = fig.add_subplot(155, projection='3d')
        else:
            ax1 = fig.add_subplot(161)
            ax2 = fig.add_subplot(162)
        ax3 = fig.add_subplot(163)
        ax4 = fig.add_subplot(164)
        ax5 = fig.add_subplot(165)
        ax6 = fig.add_subplot(166)
        fig.canvas.draw()
        fig.canvas.draw()
        plt.show(block=False)
        data = [fig, ax1, ax2, ax3, ax4,ax5,ax6]
    data[1].clear()
    data[2].clear()
    data[3].clear()
    data[4].clear()
    data[5].clear()
    data[6].clear()


    # plot the disparity
    if projection3D==True:
        data[1].plot_surface(xx, yy, disparity, rstride=1, cstride=1,
                    cmap=cm.coolwarm, linewidth=0,
                    antialiased=False)
        data[1].set_zlim3d(0,30)
    else:
        data[1].imshow(np.flipud(disparity), cmap=cm.coolwarm,vmin=0, vmax=20,
                       extent=(xx.min(), xx.max(), yy.min(),yy.max() ))
        if plotmeas==True:
            data[1].scatter(meas.T[0], meas.T[1], c=meas.T[2], s=20, 
                        cmap=cm.coolwarm)
    data[1].set_title("Depth from Stereo")

    # plot the ground truth
    if projection3D==True:
        data[2].plot_surface(xx, yy, np.flipud(GroundTruth), rstride=1, cstride=1,
                    cmap=cm.coolwarm, linewidth=0,
                    antialiased=False)
        data[2].set_zlim3d(0,30)
    else:
        data[2].imshow(np.flipud(GroundTruth), cmap=cm.coolwarm,vmin=0, vmax=20,
                       extent=(xx.min(), xx.max(), yy.min(),yy.max() ))
        if plotmeas==True:
            data[2].scatter(meas.T[0], meas.T[1], c=meas.T[2], s=30,
                        cmap=cm.coolwarm)
    
    data[2].set_title("Ground Truth")
        
    # plot the estimate
    if projection3D==True:
        data[3].plot_surface(xx, yy, mean.reshape(workspace.res,workspace.res), rstride=1, cstride=1,
                         cmap=cm.coolwarm, linewidth=0, antialiased=False)
        data[3].set_zlim3d(0,30)
    else:
        data[3].imshow(np.flipud(mean), cmap=cm.coolwarm,vmin=0, vmax=20,
                       extent=(xx.min(), xx.max(), yy.min(),yy.max() ))
        if plotmeas==True:
            data[3].scatter(meas.T[0], meas.T[1], c=meas.T[2], s=30,
                        cmap=cm.coolwarm)
    data[3].set_title("Estimate (mean)")

    # plot the variance
    # if projection3D==True:
       # data[3].plot_surface(xx, yy, sigma.reshape(workspace.res,workspace.res), rstride=1, cstride=1,
                         # cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # else:
    data[4].imshow(np.flipud(sigma), cmap=cm.coolwarm, vmin=0, vmax=1,
                       extent=(xx.min(), xx.max(), yy.min(),yy.max() ))
    if plotmeas==True:
        data[4].scatter(meas.T[0], meas.T[1], s=10,
                    cmap=cm.coolwarm)
    
    data[4].set_title("Estimate Variance")

    # adding vmin=0, vmax=1, can make this very bland
    data[5].imshow(np.flipud(aq), cmap=cm.coolwarm, 
                       extent=(xx.min(), xx.max(), yy.min(),yy.max() ))

    if plotmeas==True:
        data[5].scatter(meas.T[0], meas.T[1], s=10,
                    cmap=cm.coolwarm)

    data[5].set_title("Acquisition function")

    # plot the aquis function
    # if projection3D==True:
        # data[4].plot_surface(xx, yy, aq.reshape(workspace.res,workspace.res), rstride=1, cstride=1,
                         # cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # # else:
    # data[4].imshow(np.flipud(aq), cmap=cm.coolwarm, 
    #                extent=(xx.min(), xx.max(), yy.min(),yy.max() ))
    # if plotmeas==True:
    #     data[4].scatter(meas.T[0], meas.T[1], c=meas.T[2], s=20,
    #                     cmap=cm.coolwarm)
    # data[4].set_title("Estimate Variance")

    # plot the error
    # if projection3D==True:
        # data[5].plot_surface(xx, yy, error.reshape(workspace.res,workspace.res), rstride=1, cstride=1,
                           # cmap=cm.Greys, linewidth=0, antialiased=False)
    # else:
    data[6].imshow(np.flipud(error), cmap=cm.coolwarm,vmin=0, vmax=20,
                       extent=(xx.min(), xx.max(), yy.min(),yy.max() ))
    data[6].set_title("Error from GT")
    
    data[0].canvas.draw()
    
    if dirname is not None:
        data[0].savefig(dirname + '/' + str(iternum) + ".pdf" ,bbox_inches='tight')

    return data


def plot_belief(mean,sigma,workspace):
    # parse locations, measurements, noise from data
    xx=workspace.xx
    yy=workspace.yy
    mean=gridreshape(mean,workspace)
    sigma=gridreshape(sigma,workspace)
    fig = plt.figure(figsize=(16, 4))

    # plot the mean
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(xx, yy, mean, rstride=1, cstride=1,
                    cmap=cm.coolwarm, linewidth=0,
                    antialiased=False)
    ax.set_title("GP Mean")
        
    # plot the uncertainty
    ax1 = fig.add_subplot(122, projection='3d')
    lim=1
    cs1=ax1.plot_surface(xx, yy, sigma, rstride=1, cstride=1,
                         cmap=cm.Greys, linewidth=0, antialiased=False)
    ax1.set_title("GP Uncertainty")  

    plt.colorbar(cs1)
    plt.show()

# [errorlist,timelist,aqfunctionsnames,modelerrors


def make_error_table(fname,errtyp):
    filename = fname

    data=np.load(fname+'.npy')
    
    errordata= np.array(data[0])
    area=0.025*0.05*10000
    errordata=errordata/area*100
    timedata=np.array(data[1])
    aclabellist=np.array(data[2])
    modelerrors=np.array(data[3])

    itermax=errordata.shape[4]
    itermid=np.round(errordata.shape[4]/2.0)

    tumor=0
    iterd=itermax
    # iterd=10
    f = open(fname +errtyp+'tumor'+ ".csv", 'wb')
    titlestring='Error as a function of Gaussian noise level, for tumor' + str(tumor) + 'after iteration_'+str(iterd)#+'_noise level_'#+str(tumor1table(0,0,noiselev,0,0))
    headerstring = ",".join(aclabellist)
    
    f.write("," + titlestring+",\n")
    f.write(",," + headerstring+",\n")
    for n in range(errordata.shape[2]):
        rowstring='noise='+str(modelerrors[n])+'mean'
        rowstring2='noise='+str(modelerrors[n])+'std'

        rowdat=errordata[tumor,:,n,:,iterd-1]
        meanrowdat=rowdat.mean(axis=1)
        stdrowdat=rowdat.std(axis=1)
        f.write(","+rowstring+",{},{},{},{},{}\n".format(meanrowdat[0],meanrowdat[1],meanrowdat[2],meanrowdat[3],meanrowdat[4]))#,table[0][5]))
        f.write(","+rowstring2+",{},{},{},{},{}\n".format(stdrowdat[0],stdrowdat[1],stdrowdat[2],stdrowdat[3],stdrowdat[4]))#,table[0][5]))

    f.write("\n\n")

    titlestring='time elapsed for tumor' + str(tumor) + 'after iteration_'+str(errordata.shape[4])#+'_noise level_'#+str(tumor1table(0,0,noiselev,0,0))
    f.write("," + titlestring+",\n")
    f.write(",," + headerstring+",\n")

    for n in range(errordata.shape[2]):
        rowstring='noise='+str(modelerrors[n])
        rowdat=errordata[tumor,:,n,0,iterd-1]
        f.write(","+rowstring+",{},{},{},{},{}\n".format(meanrowdat[0],meanrowdat[1],meanrowdat[2],meanrowdat[3],meanrowdat[4]))#,table[0][5]))

    f.write("\n\n")
    iterd=itermid

    titlestring='Error as a function of Gaussian noise level, for tumor' + str(tumor) + 'after iteration_'+str(iterd)#+'_noise level_'#+str(tumor1table(0,0,noiselev,0,0))
    headerstring = ",".join(aclabellist)
    # f = open(fname +errtyp+ ".csv", 'wb')
    f.write("," + titlestring+",\n")
    f.write(",,," + headerstring+",\n")
    for n in range(errordata.shape[2]):
        rowstring='noise='+str(modelerrors[n])+'mean'
        rowstring2='noise='+str(modelerrors[n])+'std'

        rowdat=errordata[tumor,:,n,:,iterd-1]
        meanrowdat=rowdat.mean(axis=1)
        stdrowdat=rowdat.std(axis=1)
        f.write(","+rowstring+",,{},{},{},{},{}\n".format(meanrowdat[0],meanrowdat[1],meanrowdat[2],meanrowdat[3],meanrowdat[4]))#,table[0][5]))
        f.write(",,"+rowstring2+",{},{},{},{},{}\n".format(stdrowdat[0],stdrowdat[1],stdrowdat[2],stdrowdat[3],stdrowdat[4]))#,table[0][5]))
    f.write("\n\n")


    titlestring='time elapsed for tumor' + str(tumor) + 'after iteration_'+str(errordata.shape[4])#+'_noise level_'#+str(tumor1table(0,0,noiselev,0,0))
    f.write("," + titlestring+",\n")
    f.write(",," + headerstring+",\n")

    for n in range(errordata.shape[2]):
        rowstring='noise='+str(modelerrors[n])
        rowdat=errordata[tumor,:,n,0,iterd-1]
        f.write(","+rowstring+",{},{},{},{},{}\n".format(meanrowdat[0],meanrowdat[1],meanrowdat[2],meanrowdat[3],meanrowdat[4]))#,table[0][5]))

    #########
    f.write("\n\n")
    tumor=1
    iterd=itermax
    # iterd=10
    # f = open(fname +errtyp+'tumor'+ ".csv", 'wb')
    titlestring='Error as a function of Gaussian noise level, for tumor' + str(tumor) + 'after iteration_'+str(iterd)#+'_noise level_'#+str(tumor1table(0,0,noiselev,0,0))
    headerstring = ",".join(aclabellist)
    
    f.write("," + titlestring+",\n")
    f.write(",,," + headerstring+",\n")
    for n in range(errordata.shape[2]):
        rowstring='noise='+str(modelerrors[n])+'mean'
        rowstring2='noise='+str(modelerrors[n])+'std'
        rowdat=errordata[tumor,:,n,:,iterd-1]
        meanrowdat=rowdat.mean(axis=1)
        stdrowdat=rowdat.std(axis=1)
        f.write(","+rowstring+",,{},{},{},{},{}\n".format(meanrowdat[0],meanrowdat[1],meanrowdat[2],meanrowdat[3],meanrowdat[4]))#,table[0][5]))
        f.write(",,"+rowstring2+",{},{},{},{},{}\n".format(stdrowdat[0],stdrowdat[1],stdrowdat[2],stdrowdat[3],stdrowdat[4]))#,table[0][5]))

    f.write("\n\n")

    titlestring='time elapsed for tumor' + str(tumor) + 'after iteration_'+str(errordata.shape[4])#+'_noise level_'#+str(tumor1table(0,0,noiselev,0,0))
    f.write("," + titlestring+",\n")
    f.write(",," + headerstring+",\n")

    for n in range(errordata.shape[2]):
        rowstring='noise='+str(modelerrors[n])
        rowdat=errordata[tumor,:,n,0,iterd-1]
        f.write(","+rowstring+",{},{},{},{},{}\n".format(meanrowdat[0],meanrowdat[1],meanrowdat[2],meanrowdat[3],meanrowdat[4]))#,table[0][5]))

    f.write("\n\n")
    iterd=itermid

    titlestring='Error as a function of Gaussian noise level, for tumor' + str(tumor) + 'after iteration_'+str(iterd)#+'_noise level_'#+str(tumor1table(0,0,noiselev,0,0))
    headerstring = ",".join(aclabellist)
    # f = open(fname +errtyp+ ".csv", 'wb')
    f.write("," + titlestring+",\n")
    f.write(",," + headerstring+",\n")
    for n in range(errordata.shape[2]):
        rowstring='noise='+str(modelerrors[n])+'mean'
        rowdat=errordata[tumor,:,n,:,iterd-1]
        meanrowdat=rowdat.mean(axis=1)
        stdrowdat=rowdat.std(axis=1)
        f.write(","+rowstring+",,{},{},{},{},{}\n".format(meanrowdat[0],meanrowdat[1],meanrowdat[2],meanrowdat[3],meanrowdat[4]))#,table[0][5]))
        f.write(",,"+rowstring2+",{},{},{},{},{}\n".format(stdrowdat[0],stdrowdat[1],stdrowdat[2],stdrowdat[3],stdrowdat[4]))#,table[0][5]))
    f.write("\n\n")

    titlestring='time elapsed for tumor' + str(tumor) + 'after iteration_'+str(errordata.shape[4])#+'_noise level_'#+str(tumor1table(0,0,noiselev,0,0))
    f.write("," + titlestring+",\n")
    f.write(",," + headerstring+",\n")

    for n in range(errordata.shape[2]):
        rowstring='noise='+str(modelerrors[n])
        rowdat=errordata[tumor,:,n,0,iterd-1]
        f.write(","+rowstring+",{},{},{},{},{}\n".format(meanrowdat[0],meanrowdat[1],meanrowdat[2],meanrowdat[3],meanrowdat[4]))#,table[0][5]))

    f.close()


def plot_ph2_error(fname,errorslist,labels,acfunctions,modelerrors,errtyp):    
    #for e in errors:
    #         plt.plot(e)
    fig1= plt.figure(figsize=(8, 8))
    fig2=plt.figure(figsize=(8, 8))
    figs=[fig1,fig2]
    for i in range (0,errorslist.shape[0]):
        fig=figs[i]
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        ax=[ax1,ax2,ax3,ax4]
        
        # tum2rem = errorsrem[1]
        # Green: Pantone 326 C
        # RGB: [0, 178, 169], Hex: #00b2a9

        # Red/Orange: Pantone 7417 C
        # RGB: [224, 78, 57], Hex: #e04e39

        # Blue: Pantone 306 C
        # RGB [0, 181, 226],  Hex##00b5e2


        # Yellow: Pantone 123 C
        # RGB: [255, 199, 44], Hex: #ffc72c
        # gray, blue, orange, green, pink, brown, purple, yellow, red
        colors=['#4D4D4D','#5DA5DA' ,'#FAA43A' ,'#60BD68' ,'#F17CB0' ,'#B2912F' ,'#B276B2' ,'#DECF3F' ,'#F15854' ]
        #colors = ['#00b2a9','#e04e39','#00b5e2','#ffc72c','black']#blue','red','green','orange','black','cyan']
        markers=['--','-','-.','-*',':','-o']
        # for i in range(0,tum1left.shape[0]):
        #     for j in range(0,tum1left.shape[1]):
        #         expl=tum1left[i][j]
        #         expl=np.mean(expl,axis=0)
        #         #for e in range(0,tum1left.shape[2]):
        #         #    exp = tum1left[i][j][e]
        #         #    print exp.shape
        #         ax[0].plot(expl,color=colors[i])
        for k in range(0,errorslist.shape[2]):
            # for i in range(0,errorslist.shape[0]):
            for j in range(0,errorslist.shape[1]):
            
                expl=errorslist[i,j,k,:,:]
                expl=np.mean(expl,axis=0)

                ax[k].plot(expl,linewidth=2,color=colors[j])
            ax[k].set_title(errtyp+ 'level='+str(modelerrors[k]))
            ax[k].set_ylim([0,1.4])
                # ax[1].plot(expr,color=colors[i])
                # ax[2].plot(expr+expl,color=colors[i])
            if k == 3:
                ax[k].legend(acfunctions, loc='upper right')

        # ym=.08*.08
        # ym=.001
        # ax[0].set_ylim(.00, ym)
        #ax[0].set_title('error_leftover')
        # ax[1].set_ylim(.00, ym)
        # ax[1].set_title('error_removed')
        # ax[2].set_ylim(.00, ym)
        # ax[2].set_title('error_leftover+error_removed')
        plt.xlabel("Iterations")
        #ym=.08*.08
        #plt.ylim(.00, ym)

        # plt.xlim(0, 100)
        plt.ylabel("Symmetric Difference Error: cm^s")
        # plt.savefig("image_pairs/"+surface_name+'/'+name)
        # plt.close()
        plt.show()
        plt.savefig(fname + errtyp+'tumor_'+str(i)+"1.pdf", bbox_inches='tight')

    return