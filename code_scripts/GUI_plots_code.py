import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
import os
from sklearn.decomposition import PCA
#from utilities import plot_my_tree
from code_scripts.plot_tree_patch import plot_tree_patched
from code_scripts.utilities import rule_print_inline, custom_axes_limit
from matplotlib.ticker import FuncFormatter
#DPG setup
import dearpygui.dearpygui as dpg
from dearpygui_ext.themes import create_theme_imgui_light
from code_scripts.utilities import colormap_from_str


class interactable_point: #Object containg all information of a point
    def __init__(self,name,pos,color,size,shape,cluster_memb=None,value=None):
        self.name=str(name)                 #string 
        self.pos=pos                        #list or tuple with two floats
        self.color=color                    #list or tuple with 4 integers from 0 to 255 (rgba)
        #self.edges=edges                   #string (default: "black"), customisation not implemented
        self.size=size                      #positive integer
        self.shape=shape                    #string
        self.cluster_memb=cluster_memb      #string or None
        self.value=value                    #float or None (learner predicted value)
    
class interactable_plot: #Object containing all information of a plot
    def __init__(self,name,points,clustered=False,xlabel="PC1",ylabel="PC2"):
        self.name=str(name)                 #string (should be unique)
        self.points=points                  #list of interactable_points objects
        self.clustered=clustered            #Boolean
        self.xlabel=xlabel                  #string
        self.ylabel=ylabel                  #string 


          
def make_interactive_plot(plots, plotheight=400,borderpercent=0.1,
                          other_inputs=None,
                          max_depth=None): #Function that makes the interctive plots
    

    dpg.destroy_context()
    dpg.create_context()
      
    #DPG set theme
    lighttheme = create_theme_imgui_light()
    dpg.bind_theme(lighttheme)
    
    #Load and set up tree image
    # create image to load:
    
    # Activated when mouse is left clicked on a DPG object
    # only at this stage, the tree index is known,
    # as it corresponds to the point.hovered at the moment
    def mouse_click_left_callback():
        
        #Check all points to see if they are hovered
        for plot in plots:
            for point in plot.points:
                if point.hovered:
                    
                    if other_inputs is not None: # extra inputs are given to the user!
                    
                        # the other_input is supposed to be a TreeExtractor instance
                    
                        my_clf = other_inputs.clf # rf estiomator here
                        feature_names = my_clf.feature_names_in_
                        sample_index = other_inputs.sample.index[0]
                        #node_indicator = my_clf.decision_path(other_inputs.sample)
                        #leaf_id = my_clf.apply(other_inputs.sample)                        
                        tree_index = int(point.name)
                        
                        rule_print_inline(my_clf[tree_index], other_inputs.sample)
                        # print_inline is not adapted to rf
                        # TODO, prints in console, to change
                        tree_name_png = "Tree_" + str(tree_index) + ".png"
                        # plot_my_tree(my_clf, tree_index, feature_names,
                        #             sample_index, tree_name_png)
                        
                        print("Tree plot {} for sample {}:".format(tree_index, sample_index))
                        
                        if max_depth != None:
                            real_plot_leaves = max(my_clf[tree_index].tree_.n_leaves, 0.75*(2**max_depth))
                            real_plot_depth = min(my_clf[tree_index].tree_.max_depth, max_depth)

                        real_plot_leaves = my_clf[tree_index].tree_.n_leaves
                        real_plot_depth = my_clf[tree_index].tree_.max_depth
    
                    
                        smart_width = 1 + 1.0*np.sqrt(real_plot_leaves**1.5)
                        smart_width = int(smart_width)
                        
                        smart_height = real_plot_depth*0.9
                        smart_height = int(smart_height)
                        #print("figsize:", (smart_width, smart_height))
                        fig, ax = plt.subplots(figsize=(smart_width, smart_height))
                        #plt.rcParams["figure.figsize"] = (smart_width, smart_height)
                        
                        
                        with warnings.catch_warnings(): #TODO it is not catching the warning! search better
                            warnings.simplefilter("ignore")
                            
                            the_tree = my_clf[tree_index]
                            
                                                    
                            plot_tree_patched(the_tree,
                                      max_depth=max_depth,
                                      feature_names=feature_names,
                                      fontsize=12)
                            plt.rcParams["font.size"] = 14 
                            plt.title("Tree %i predicting sample %i" % (tree_index, sample_index))
                            plt.savefig(tree_name_png)
                            plt.show()

                        # not that the needed tree image is created, we can load it

                        # generates provisional plot
                        # load tree_name_png, or  "trial"
                        IMwidth, IMheight, IMchannels, IMdata = dpg.load_image(tree_name_png) #"PL"+plot.name+"_"+point.name+".png"
                        #print("done, consider destroying the point")
                        # deleting Tree_.png file
                        os.remove(os.path.join(tree_name_png))
                   
                    
                    elif other_inputs is None: #other_inputs is None
                        IMwidth, IMheight, IMchannels, IMdata = dpg.load_image("test3.png") #"PL"+plot.name+"_"+point.name+".png"
                        print("done, consider destroying the point")
                    
                    with dpg.texture_registry(show=False):
                        treeplot=dpg.add_static_texture(IMwidth, IMheight, IMdata)
                        
                    # Opens window showing tree
                    popup_window=dpg.add_window(width=min(IMwidth+4*FULLSPACING, windowwidth),
                                                height=min(IMheight+5*FULLSPACING, windowheight),
                                                modal=True,
                                                horizontal_scrollbar=True)
                    
                    dpg.add_image(treeplot, width=IMwidth,
                                  height=IMheight,
                                  parent=popup_window)
    
    #DPG mouse click setup 
    with dpg.handler_registry():
        dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left,
                                    callback=mouse_click_left_callback)
        
    
    # Activated every frame 
    def point_draw_and_interact(sender,app_data,user_data):
        
        #Calculates positions in pixel space
        _helper_data = app_data[0]
        transformed_x = app_data[1]
        transformed_y = app_data[2]
        mouse_x_pixel_space = _helper_data["MouseX_PixelSpace"]
        mouse_y_pixel_space = _helper_data["MouseY_PixelSpace"]
        
        #Renaming for legibility
        plot=user_data
        points=plot.points
        
        #Deletes the previous frame
        dpg.delete_item(sender, children_only=True, slot=2) #Slot=2 specifies that only the dpg.draw objects must be deleted.
        dpg.configure_item(sender, tooltip=False)
        
        #Goes through all points to draw them and to see if they are hovered
        for i, point in enumerate(points):
            
            # Draws point shape
            if point.shape=="circle":
                dpg.draw_circle([transformed_x[i], transformed_y[i]],
                                point.size,fill=point.color,parent=sender)
            elif point.shape=="triangle":

                sin_size = 0.500*point.size
                cos_size = 0.866*point.size

                dpg.draw_triangle([transformed_x[i],
                                   transformed_y[i]-point.size],
                                  [transformed_x[i]+cos_size,transformed_y[i]+sin_size],
                                  [transformed_x[i]-cos_size,transformed_y[i]+sin_size],
                                  fill=point.color,parent=sender)
            elif point.shape=="square":
                half_side_length=0.707*point.size
                dpg.draw_rectangle([transformed_x[i]-half_side_length,
                                    transformed_y[i]-half_side_length],
                                   [transformed_x[i]+half_side_length,
                                    transformed_y[i]+half_side_length],
                                   fill=point.color,parent=sender)
            else:
                ValueError("Shape {} not supported".format(point.shape))
                
            # Checks if cursor close enough for point to be considered hovered
            x_dif=transformed_x[i]-mouse_x_pixel_space
            y_dif=transformed_y[i]-mouse_y_pixel_space
            norm_squared=x_dif**2+y_dif**2
            if norm_squared<= point.size**2:
                point.hovered=True
                dpg.configure_item(sender, tooltip=True)
                
                #Adds extra information to tooltip depending on point type
                tool_tip_text="hovered Point: " + point.name
                if point.cluster_memb is not None:
                    tool_tip_text+="\n"+"Cluster: " +point.cluster_memb
                if point.value is not None:
                    tool_tip_text+="\n"+"Value: " +point.value
                    
                #Shows tooltip when point is hovered
                dpg.set_value("TooltipText"+plot.name,tool_tip_text)
            else:
                point.hovered=False

    
    #Function that draws the average position of the different clusters     
    def cluster_average(sender,app_data):
        size=9
        x = app_data[1]
        y = app_data[2]
        
        #Reset previous frame
        dpg.delete_item(sender, children_only=True, slot=2) #Deletes all drawn items (draw = slot 2)
        
        #Draws the average position
        for i in range(len(x)):
            dpg.draw_circle([x[i],y[i]],size, fill=[150,150,150,255],parent=sender)
            dpg.draw_text([x[i]-size/2,y[i]-size],clusters[i],size=size*2,parent=sender)
            
    # A few useful variables
    plotamount=len(plots)
    FULLSPACING=8 #This is a constant used for DPG positioning (standard distance between two elements or between an element and the border of the parent container)

    
    # Loads in the colourbar images, saving the desired size.
    addedwidth=0
    for plot in plots:
        
        #Loads and sets up the image
        IMwidth, IMheight, IMchannels, IMdata = dpg.load_image('colourbar'+plot.name+'.png')
        with dpg.texture_registry(show=False):
            dpg.add_static_texture(IMwidth,IMheight,IMdata,tag="colourbar"+plot.name)
        
        #Calculates the wanted width for the image to preserve spect ratio and saves it to a total for calculating the nessesary window size
        plot.colour_bar_width=round(plotheight*IMwidth/IMheight)
        addedwidth+=plot.colour_bar_width

    # Calculates window sizes based on plot size/amount
    windowwidth=plotamount*plotheight+(2*plotamount+1)*FULLSPACING+addedwidth
    windowheight=plotheight+2*FULLSPACING
    
    # Calculates x- and y-axis limits based on points
    for plot in plots:
        x_values=[]
        y_values=[]
        for point in plot.points:
            x_values.append(point.pos[0])
            y_values.append(point.pos[1])
        x_diff=max(x_values)-min(x_values)
        y_diff=max(y_values)-min(y_values)
        plot.x_axis_limits=[min(x_values)-x_diff*borderpercent,
                            max(x_values)+x_diff*borderpercent]
        plot.y_axis_limits=[min(y_values)-y_diff*borderpercent,
                            max(y_values)+y_diff*borderpercent]

    # Calculates cluster averages if there are clusters
    for plot in plots:
        if plot.clustered: # only on the left plot (where clustering is shown)
            
            #Gets the names of all the different clusters in the plot
            clusters=[]
            for point in plot.points:
                if point.cluster_memb not in clusters:
                    clusters.append(point.cluster_memb)
                    
            #gets the average coordinates of all the clusters
            averagecoords=[]
            for cluster in clusters:
                coords=[]
                for point in plot.points:
                    if point.cluster_memb==cluster:
                        coords.append(point.pos)
                averagecoord = [sum(x)/len(x) for x in zip(*coords)]
                averagecoords.append(averagecoord)
        
    sample_index = other_inputs.sample.index[0]

    #DPG build windows
    with dpg.window(tag="MainWindow", width=windowwidth,
                    label="Explaining sample "+ str(sample_index) + ", close the window to go to the next sample",
                    height=windowheight, no_title_bar=False,
                    no_move=True,no_resize=True):
        with dpg.group(horizontal=True):
            for plot in plots:
                # Adds interactable plot
                with dpg.plot(tag="Plot"+plot.name,height=plotheight,width=plotheight):
                    
                    # Creates x axis
                    dpg.add_plot_axis(dpg.mvXAxis, label=plot.xlabel, tag="x_axis"+plot.name)
                    dpg.set_axis_limits("x_axis"+plot.name, plot.x_axis_limits[0], plot.x_axis_limits[1])
                    
                    # Creates y axis
                    dpg.add_plot_axis(dpg.mvYAxis, label=plot.ylabel, tag="y_axis"+plot.name)
                    dpg.set_axis_limits("y_axis"+plot.name,plot.y_axis_limits[0],plot.y_axis_limits[1])
                    
                    # Adds points to correct plot
                    pointcoordx=[point.pos[0] for point in plot.points]
                    pointcoordy=[point.pos[1] for point in plot.points]
                    with dpg.custom_series(pointcoordx,pointcoordy,2, user_data=plot,callback=point_draw_and_interact,parent="y_axis"+plot.name,tooltip=True):
                        dpg.add_text("Placeholder",tag="TooltipText"+plot.name) #You can add other objects in here instead of text if you want!
                    
                    # Adds center of clusters
                    if plot.clustered:
                        dpg.add_custom_series([coord[0] for coord in averagecoords],[coord[1] for coord in averagecoords],2,callback=cluster_average,parent="y_axis"+plot.name)
                          
                # Adds colourbar image next to plot        
                dpg.add_image('colourbar'+plot.name,height=plotheight, width=plot.colour_bar_width)
                
                
    #DPG make viewport
    viewport=dpg.create_viewport(title='Plot', width=800, height=600)
    dpg.configure_viewport(viewport,height=windowheight+5*FULLSPACING,width=windowwidth+2*FULLSPACING)
    
    #DPG show app
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.render_dearpygui_frame()
    dpg.destroy_context()


    
def plot_with_interface(plot_data_bunch, kmeans,
                           input_method, # a fitted bellatrex instance
                           max_depth=None,
                           colormap=None,
                           clusterplots=[True,False]):

    def shaper(in_shape):
        if in_shape is True:
            return "triangle"
        else:
            return "circle"
        
    def sizer(in_size):
        if in_size is True:
            return 9.0
        else:
            return 4.5
    def edger(in_edge):
        return "black"
        # else:
        #     return (1,1,1,0.5) # semi-transparent white
    
    def rgbaconv(mpl_rgba):
        dpg_rgba=[i*255 for i in mpl_rgba]
        return dpg_rgba
    
    
    # Custom formatter function for colorabar on ax4
    # not working correctly..
    def custom_formatter(x, pos): # pos paramter to comply with expected signature
        if np.abs(x) < 1e-7: # 0.00 for near zero values
            return f"{x:.2f}"
        if 1e-2 <= np.abs(x) < 1:
            return f"{x:.2f}"  # 2 decimal digits for numbers between -1 and 1
        elif 1 <= np.abs(x) < 10:
            return f"{x:.1f}"  # 1 decimal digit 
        elif 10 <= np.abs(x) < 100:
            return f"{x:.0f}"  # 0 decimal digits (round to nearest integer)
        else: # 1e-7 < np.abs(x) < 1e-2 or  np.abs(x) > 100
            return f"{x:.1e}"  # Scientific notation with 2 significant digits
    
    #repeat PCA to 2 dimensions for projected trees
    #(original proj dimension can be > 2)
    # better: use original one and keep only the first two dims.
    PCA_fitted = PCA(n_components=2).fit(plot_data_bunch.proj_data)
    plottable_data = PCA_fitted.transform(plot_data_bunch.proj_data)  # (lambda,2)

    #centers = PCA_fitted.transform(kmeans.cluster_centers_)
    cluster_memb = kmeans.labels_
    
    final_ts_idx = input_method.final_trees_idx
    
    is_final_candidate = [plot_data_bunch.index[i] in final_ts_idx for i in range(len(plot_data_bunch.index))]
    custom_sizes = list(map(sizer, is_final_candidate))
    custom_shapes = list(map(shaper, is_final_candidate))
    #custom_edges = list(map(edger, is_final_candidate))
    
    plots=[]
    
    color_map_left = plt.cm.viridis  # default colormap for clustering plot
    color_map_left = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', 
                                    [color_map_left(i) for i in range(color_map_left.N)], color_map_left.N)
    
    color_map_right = colormap_from_str(colormap) # User custom colormap from object or from string
  
    # colormap = color_map_left if clustered == True else color_map_right
    
    for plotindex, clustered in enumerate(clusterplots):
        
        if clustered:
            fig,ax= plt.subplots(1, 1, figsize=(1, 4.5), dpi=100)

            # define the bins and normalize
            freqs = np.bincount(cluster_memb)
            if np.min(freqs) == 0:
                 raise KeyError("There are empty clusters, the scatter and colorbar could differ in color shade")
            norm_bins = list(np.cumsum(freqs))
            norm_bins.insert(0, 0)
            
            # transform list to array (ticks location needs arithmentic computation)
            norm_bins = np.array(norm_bins)
            
            # create label names
            labels = []
            for i in np.unique(cluster_memb):
                labels.append("cl.{:d}".format(i+1))
            # normalizing color, prepare ticks, labels
            norm = mpl.colors.BoundaryNorm(norm_bins, color_map_left.N)
            tickz = norm_bins[:-1] + (norm_bins[1:] - norm_bins[:-1]) / 2
                
            cb1 = mpl.colorbar.Colorbar(ax, cmap=color_map_left, norm=norm,
                 spacing='proportional', ticks=tickz, boundaries=norm_bins, format='%1i')
                 #label="cluster membership")
            cb1.ax.set_yticklabels(labels)  # vertically oriented colorbar

            ax.yaxis.set_ticks_position('left')
            norms=[norm(norm_bins[cluster_memb[i]]) for i in range(len(cluster_memb))]
            
        else: # not the clustering plot, but the prediction/loss distribution:
            fig, ax= plt.subplots(1, 1, figsize=(1.2, 4.5), dpi=100) 
            
            # binary, regression, survival case (single-output cases)
            if isinstance(plot_data_bunch.RF_pred, float) or plot_data_bunch.RF_pred.size == 1:
                
                is_binary = (plot_data_bunch.set_up == "bin")
                
                plot_data_bunch.RF_pred = np.array(plot_data_bunch.RF_pred).squeeze()
                
                v_min, v_max = custom_axes_limit(np.array(plot_data_bunch.pred).min(),
                                  np.array(plot_data_bunch.pred).max(),
                                  plot_data_bunch.RF_pred,
                                  is_binary)
                
                norm_preds = mpl.colors.BoundaryNorm(np.linspace(v_min,v_max, 256),
                                                      color_map_right.N)
                
                ## add to colorbar a line corresponding to Bellatrex prediction                
                pred_tick = np.round(plot_data_bunch.RF_pred,3)
              
                cb2 = mpl.colorbar.Colorbar(ax, cmap=color_map_right, norm=norm_preds,
                                            label="RF pred: " + str(pred_tick))
                
                plot_data_bunch.pred = np.array(plot_data_bunch.pred).squeeze() # force shape: (n_trees,)
                
                # add tick in correspondence to the single trees prediction
                cb2.ax.plot([0, 1], [plot_data_bunch.pred]*2, color='grey',
                            linewidth=1)

                # add indicator (visually: >--<) in correspondence to the ensemble prediction
                cb2.ax.plot([0.02, 0.98], [pred_tick]*2, color='black', linewidth=2.5, marker="P")
            
            # multi-output case, L2 losses instead of predictions
            else:
                # Blue for small losses, red for big losses
                # color_map_right = colormap_from_str('RdYlBu_r')
                color_map_right = colormap_from_str(colormap) # User custom colormap from object or from string


                v_min, v_max = custom_axes_limit(np.array(plot_data_bunch.loss).min(),
                                  np.array(plot_data_bunch.loss).max(),
                                  force_in=None,
                                  is_binary=False)
                
                
                norm_preds = mpl.colors.BoundaryNorm(np.linspace(v_min,v_max, 256),
                                                      color_map_right.N)
                
                
                cb2 = mpl.colorbar.Colorbar(ax, cmap=color_map_right, norm=norm_preds,
                                            label=str(input_method.fidelity_measure)+' loss')
                cb2.ax.plot([0, 1], [plot_data_bunch.loss]*2, color='grey',
                            linewidth=1)
            
            # ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2g'))            
            ticks_to_plot = ax.get_yticks()
            
            if np.abs(np.min(ticks_to_plot)) < 1e-3 and np.abs(np.max(ticks_to_plot)) > 1e-2:
                min_index = np.argmin(ticks_to_plot)
                ticks_to_plot[min_index] = 0
                ax.set_yticks(ticks_to_plot)

            ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))
            ax.minorticks_off()
            
            # single output case: float, or (1,)-shaped ndarray
            if isinstance(plot_data_bunch.RF_pred, float) or plot_data_bunch.RF_pred.size == 1 : 
                norms=[norm_preds(float(plot_data_bunch.pred[i])) for i in range(len(cluster_memb))]
            else: # multi output case
                norms=[norm_preds(plot_data_bunch.loss[i]) for i in range(len(cluster_memb))]


        fig.tight_layout()
        fig.savefig('colourbar'+str(plotindex))
        
        # use appropriate colormap (right plot vs left plot)
        cmap_gui = color_map_left if clustered else color_map_right
        
        colours = [rgbaconv(cmap_gui(norms[i])) for i in range(len(plot_data_bunch.index))]
        
        # DEBUG: visualise plotted colors
        # plt.scatter(range(100), range(100), c=np.array(colours)/255, s=10)
        # plt.show()
        
        # Here the GUI objects are instantiated: the consist in a collection of
        # interactable_point classes
    
        points=[]
        for j in range(len(plot_data_bunch.index)):
            points.append(interactable_point(plot_data_bunch.index[j], #name
                                             plottable_data[j], #pos
                                             colours[j], #color
                                             custom_sizes[j], #size
                                             #custom_edges[j], #edge
                                             custom_shapes[j]) #shape
                                             )
            
            # adds extra information to the point for the tooltip
            if clustered:   
                points[j].cluster_memb=str(cluster_memb[j]+1) #+1 to change from 0-indexed to 1-indexed
            else:
                if isinstance(plot_data_bunch.pred[j], float):
                    points[j].value=f'{plot_data_bunch.pred[j]:.4f}' #Show 4 numbers after decimal point
                elif isinstance(plot_data_bunch.loss[j], float):
                    points[j].value=f'{plot_data_bunch.loss[j]:.4f}' #Show 4 numbers after decimal point
                else:
                    ValueError("expecting float, got {} instead".format(type(plot_data_bunch.loss)))
        plots.append(interactable_plot(plotindex, points,clustered=clustered))
    
    # end of the enumerate(clusterplots) `for' loop (length 2 -> two plots)
    make_interactive_plot(plots,
                          plotheight=400,
                          borderpercent=0.1,
                          other_inputs=input_method,
                          max_depth=max_depth)
    
    plt.show()
    
    return
