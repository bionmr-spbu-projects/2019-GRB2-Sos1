#!/usr/bin/python2.7

#
# WARNING: No other global imports are allowed here, use imports inside class functions instead
#          Current implementation can not track such dependencies and your run will fail

from runmd2.MD import MD


class MD_grb2(MD):

    def __init__(self, init_struct, final_struct):
        import os
        from pyxmol import readPdb
        restricted_structure = readPdb(init_struct)[0]
        final_structure = readPdb(final_struct)[0]
        self.path_to_final_struct = os.path.abspath(final_struct)
        wd = "run12" # os.path.splitext(os.path.basename(init_struct))[0]
        super(MD_grb2, self).__init__(name=wd, trj_home_dir=wd)

        restricted_structure.writeAsPdb(open("input.pdb", "w"))
        self.save_state("saved_state.pickle")
        final_structure.writeAsPdb(open("input2.pdb", "w"))

    def run_setup(self):
        import os
        from pyxmol import readPdb
        from pyxmol.predicate import rId
        #from pyxmolpp import Trajectory, CrystalRestorer, make_lattice_vectors, FrameShellBuilder 


        self.log("Setup > ")
        self.tleaprc.source(os.environ["PRMTOP_HOME"]+"/cmd/leaprc.protein.chlorolys.ff14SB")
        self.tleaprc.source(os.environ["PRMTOP_HOME"]+"/cmd/leaprc.water.tip3p")        
        self.tleaprc.add_command("loadamberparams frcmod.ionsjc_tip3p")
        self.tleaprc.load_pdb("./input.pdb")
        self.tleaprc.solvate_oct("TIP3PBOX", 10.0)
        self.tleaprc.add_ions("Na+",target_charge=0)
        self.tleaprc.save_params(output_name=MD._build_dir+"/box")
        self.tleaprc.save_pdb(output_name=MD._build_dir+"/box")
        self.tleaprc.add_command("quit")
        
        self.n_residues = n_residues = 68
        
        debug_factor = 1
        
        self.min1_parameters.set(
            imin=1,
            maxcyc=500,
            ncyc=100,
            ntb=1,
            ntr=1,
            cut=10.0
        ).add_atom_pin(200, None, [(1,n_residues)])
        self.min2_parameters.set(
            imin=1,
            maxcyc=100,
            ncyc=200,
            ntb=1,
            ntr=0,
            cut=10.0
        ).add_atom_pin(0, None, [(1,n_residues)])
        self.heat_parameters.set(
            imin=0,
            irest=0,
            ntx=1,
            ntb=1,
            ntr=1,
            ntc=2,
            ntf=2,
            tempi=0.0,
            temp0=298.0,
            ntt=1,
            nstlim=10000,
            dt=0.001,
            ntpr=50,
            ntwx=50,
            ntwr=50,
            ioutfm=1
        ).add_atom_pin(0, None, [(1,n_residues)])
        self.equil_parameters.set(
            imin=0,
            irest=1,
            ntx=5,
            ntb=2,
            iwrap=1,
            ntt=3,
            gamma_ln=2.0,
            ig=-1,
            tempi=298.0,
            temp0=298.0,
            ntp=1,
            pres0=1.0,
            taup=2.0,
            cut=10.0,
            ntr=1,
            ntc=2,
            ntf=2,
            nstlim=500000/debug_factor,
            dt=0.002,
            ntpr=500,
            ntwx=500,
            ntwr=500000/debug_factor,
            ioutfm=1
        )        
 
        self.run_parameters.set(
            imin=0,
            irest=1,
            ntx=5,
            ntb=2,
            iwrap=1,
            ig=-1,
            ntt=3,
            gamma_ln=2.0,
            tempi=298.0,
            temp0=298.0,
            ntp=1,
            pres0=1.0,
            taup=2.0,
            cut=10.5,
            ntr=0,
            ntc=2,
            ntf=2,
            nstlim=500000/debug_factor, 
            dt=0.002,
            ntpr=500,
            ntwx=500,
            ntwv=0,
            ntwr=500000/debug_factor,
            ioutfm=1
        )
        

        self._pmemd_executable = ["pmemd.cuda"]

        self.build()
        
        self.prepare_prefinal_topology()
        self.prepare_final_topology()
        
        self.restricted_structure = readPdb(self.tleaprc.pdb_output_name+".pdb")[0].asResidues >> (rId<=self.n_residues)
        self.minimize()
        self.heat()
        self.equilibrate()

        self.setup_is_done = True
        
        self.save_state("saved_state.pickle")
        self.log("Setup < ")
    
    def prepare_prefinal_topology(self):
        import os
        # self.path_to_final_struct
        from runmd2.MD import TleapInput
    
        from parmed.amber import AmberParm, AmberMask
        from parmed.topologyobjects import Atom
    
        
        final_tleap = TleapInput()
        final_tleap.source(os.environ["PRMTOP_HOME"]+"/cmd/leaprc.protein.chlorolys.ff14SB")
        final_tleap.source(os.environ["PRMTOP_HOME"]+"/cmd/leaprc.water.tip3p")        
        final_tleap.add_command("loadamberparams frcmod.ionsjc_tip3p")
        final_tleap.load_pdb("./input2.pdb")
        #final_tleap.add_command("bond wbox.32.SG wbox.69.C2 ")
        final_tleap.solvate_oct("TIP3PBOX", 15.0)
        final_tleap.add_ions("Na+",target_charge=0)
        final_tleap.save_params(output_name=MD._build_dir+"/prefinal_box")
        final_tleap.save_pdb(output_name=MD._build_dir+"/prefinal_box")
        final_tleap.add_command("quit")
        
        final_tleap.write_commands_to(MD._build_dir+"/prefinal_tleap.rc")
        
        self.call_cmd(["tleap", "-s", "-f", MD._build_dir+"/prefinal_tleap.rc"])
        
        
        # strip redundant water molecules 
        parm_old = AmberParm(MD._build_dir+"/box.prmtop")
        parm_new = AmberParm(MD._build_dir+"/prefinal_box.prmtop")
        old_res = set()
        new_res = set()
        for r in parm_old.residues:
            old_res.add(r.number)
            number = r.number
        old_res.add(number+1)
        for r in parm_new.residues:
            new_res.add(r.number)
        residues_to_delete = sorted(list(new_res - old_res))
        res_mask = ":"+','.join(list(map(str, residues_to_delete)))
        parm_new.strip(res_mask)
        parm_new.write_parm(MD._build_dir+"/prefinal_box.mod.prmtop")
        
        
    def prepare_final_topology(self):
        import os 
        # self.path_to_final_struct
        from runmd2.MD import TleapInput
    
        from parmed.amber import AmberParm, AmberMask
        from parmed.topologyobjects import Atom
    
        
        final_tleap = TleapInput()
        final_tleap.source(os.environ["PRMTOP_HOME"]+"/cmd/leaprc.protein.chlorolys.ff14SB")
        final_tleap.source(os.environ["PRMTOP_HOME"]+"/cmd/leaprc.water.tip3p")        
        final_tleap.add_command("loadamberparams frcmod.ionsjc_tip3p")
        final_tleap.load_pdb("./input2.pdb")
        final_tleap.add_command("bond wbox.32.SG wbox.68.C2 ")
        final_tleap.solvate_oct("TIP3PBOX", 15.0)
        final_tleap.add_ions("Na+",target_charge=0)
        final_tleap.save_params(output_name=MD._build_dir+"/final_box")
        final_tleap.save_pdb(output_name=MD._build_dir+"/final_box")
        final_tleap.add_command("quit")
        
        final_tleap.write_commands_to(MD._build_dir+"/final_tleap.rc")
        
        self.call_cmd(["tleap", "-s", "-f", MD._build_dir+"/final_tleap.rc"])
        
        
        # strip redundant water molecules 
        parm_old = AmberParm(MD._build_dir+"/box.prmtop")
        parm_new = AmberParm(MD._build_dir+"/final_box.prmtop")
        old_res = set()
        new_res = set()
        for r in parm_old.residues:
            old_res.add(r.number)
            number = r.number
        old_res.add(number+1)
        for r in parm_new.residues:
            new_res.add(r.number)
        residues_to_delete = sorted(list(new_res - old_res))
        res_mask = ":"+','.join(list(map(str, residues_to_delete)))
        parm_new.strip(res_mask)
        parm_new.write_parm(MD._build_dir+"/final_box.mod.prmtop")
    
    def modify_prefinal_topology(self):
        from copy import deepcopy
        new_topology_mod = deepcopy(self.parmed)
        old_topology_mod = self.parmed
        self.parmed = new_topology_mod
        self.log("change VdW interactions with CL")
        self.parmed.add_command('changeLJSingleType :Cl-@Cl- {radius} {depth}'.format(radius = 1.948, depth = 0.265))
        self.log("change charges")
        for command in ["change CHARGE :LYZ@CD {charge}".format(charge = 0.172303),
            "change CHARGE :LYZ@HD2 {charge}".format(charge = -0.064421),
            "change CHARGE :LYZ@HD3 {charge}".format(charge = -0.064421),
            "change CHARGE :LYZ@CE {charge}".format(charge = 0.161613),
            "change CHARGE :LYZ@HE2 {charge}".format(charge = 0.007692),
            "change CHARGE :LYZ@HE3 {charge}".format(charge = 0.007692),
            "change CHARGE :LYZ@NZ {charge}".format(charge = -0.620004),
            "change CHARGE :LYZ@HZ {charge}".format(charge = 0.342511),
            "change CHARGE :LYZ@C1 {charge}".format(charge = 0.846209),
            "change CHARGE :LYZ@OZ {charge}".format(charge = -0.619871),
            "change CHARGE :LYZ@C2 {charge}".format(charge = -0.558636),
            "change CHARGE :Cl-@Cl- {charge}".format(charge = -0.136147),
            "change CHARGE :LYZ@H1 {charge}".format(charge = 0.253025),
            "change CHARGE :LYZ@H2 {charge}".format(charge = 0.253025),
            "change CHARGE :CYZ@CA {charge}".format(charge = -0.0351),
            "change CHARGE :CYZ@HA {charge}".format(charge = 0.0508),
            "change CHARGE :CYZ@CB {charge}".format(charge = -0.2413),
            "change CHARGE :CYZ@HB2 {charge}".format(charge = 0.1122),
            "change CHARGE :CYZ@HB3 {charge}".format(charge = 0.1122),
            "change CHARGE :CYZ@SG {charge}".format(charge = -0.8844),
            ]:
            self.log(command)
            self.parmed.add_command(command)
        for command in ['setBond :CYZ@CB :CYZ@SG 237 1.81',
                        'deleteDihedral :CYZ@C :CYZ@CA :CYZ@CB :CYZ@SG',
                        'deleteDihedral :CYZ@N :CYZ@CA :CYZ@CB :CYZ@SG',
                        'addDihedral :CYZ@C :CYZ@CA :CYZ@CB :CYZ@SG {k} 3 0 1.2 2 type "normal"'.format(k = 0.1556),
                        'addDihedral :CYZ@N :CYZ@CA :CYZ@CB :CYZ@SG {k} 3 0 1.2 2 type "normal"'.format(k = 0.1556)
            ]:
            self.log(command)
            self.parmed.add_command(command)
        self.log("Commands from parmed0")
        for command in self.parmed.get_commands():
            self.log("Command: {}".format(command))
        self.log("Finish printing commands0")
        self.write_mod_topology()
        self.write_mod_topology(outname='./1_build/box_before_lock_mod.prmtop')
        self.parmed = old_topology_mod
    
    def prepare_final_rst(self, parm_old):
        from parmed.amber import AmberParm, Rst7
        from copy import deepcopy
        
        rst = Rst7(MD._run_dir+"/run"+MD._pattern%self.current_step+"mod.rst")
        parm_old.load_rst7(rst)
        for r in parm_old.residues:
            if r.name == 'LYC':
                # print(r.number)
                lyc = r
                for a in r:
                    if a.name == 'CL':
                        new_Cl = deepcopy(a)
                        # print(r.name, r.number, a.name, a.idx, a.xx, a.xy, a.xz, a.vx, a.vy, a.vz)
        parm_old.strip('@CL')
        parm_old.add_atom_to_residue(new_Cl, lyc)
        parm_old.write_rst7(MD._run_dir+"/run"+MD._pattern%self.current_step+"mod.mod.rst")
        self._restart_filename = MD._run_dir+"/run"+MD._pattern%self.current_step+"mod.mod.rst"
    
    def make_current_lattice_vectors(self, box): 
        import numpy as np
        a, b, c, alpha, beta, gamma = box[0], box[1], box[2], box[3], box[4], box[5]    
        alpha, beta, gamma = np.radians(alpha), np.radians(beta), np.radians(gamma)
        A = np.array([a, 0.0, 0.0])
        B  = np.array([b*np.cos(gamma), b * np.sin(gamma),0 ])
        c0 = c*np.cos(beta);
        c1 = b/B[1] * c * np.cos(alpha) - B[0]/B[1]*c0;
        c2 = np.sqrt(c*c - c0*c0 - c1*c1)
        C = np.array([c0, c1, c2])
        return A, B, C
    
    
    def get_shift_from_nc(self, frame_number, res1_number, atom1_name, res2_number, atom2_name, parm, traj):
        from pyxmol.geometry import distance, angle_deg, translate
        from parmed.amber import NetCDFTraj, Rst7, AmberParm 
        from parmed.geometry import distance2
        import numpy as np 
        
        
        new_rst7 = Rst7(natom=traj.atom)
        new_rst7.coordinates = traj.coordinates[frame_number]
        new_rst7.box = traj.box[frame_number]
       
        parm.load_rst7(new_rst7)
        for a in parm.atoms:
            if  a.residue.number == (res1_number-1) and a.name == atom1_name:
                A1 = a   
            elif a.residue.number == (res2_number-1) and a.name == atom2_name:
                A2 = a
        best_dist = np.sqrt(distance2(A1, A2))
        A, B, C = self.make_current_lattice_vectors(traj.box[frame_number]) 
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    shift = i*A + j*B + k*C
                    A2.xx += shift[0]
                    A2.xy += shift[1]
                    A2.xz += shift[2] 
                    dist = np.sqrt(distance2(A1, A2))
                    if dist < best_dist:
                        best_dist = dist
                        best_shift = shift 
                    A2.xx -= shift[0]
                    A2.xy -= shift[1]
                    A2.xz -= shift[2] 
        return best_dist, best_shift
    
    def get_shift_from_nc_2(self, frame_number, res1_number, atom1_name, res2_number, atom2_name, parm, traj, res3_number, atom3_name, res4_number, atom4_name):
        from pyxmol.geometry import distance, angle_deg, translate
        from parmed.amber import NetCDFTraj, Rst7, AmberParm 
        from parmed.geometry import distance2
        import numpy as np 
        
        
        new_rst7 = Rst7(natom=traj.atom)
        new_rst7.coordinates = traj.coordinates[frame_number]
        new_rst7.box = traj.box[frame_number]
       
        parm.load_rst7(new_rst7)
        for a in parm.atoms:
            if  a.residue.number == (res1_number-1) and a.name == atom1_name:
                A1 = a   
            elif a.residue.number == (res2_number-1) and a.name == atom2_name:
                A2 = a
            elif a.residue.number == (res3_number-1) and a.name == atom3_name:
                A3 = a
            elif a.residue.number == (res4_number-1) and a.name == atom4_name:
                A4 = a
        check_dist = np.sqrt(distance2(A1, A2))
        best_shift = np.array([0, 0, 0])
        A, B, C = self.make_current_lattice_vectors(traj.box[frame_number]) 
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    shift = i*A + j*B + k*C
                    A2.xx += shift[0]
                    A2.xy += shift[1]
                    A2.xz += shift[2] 
                    dist = np.sqrt(distance2(A1, A2))
                    if dist <= check_dist:
                        check_dist = dist
                        best_shift = shift 
                    A2.xx -= shift[0]
                    A2.xy -= shift[1]
                    A2.xz -= shift[2] 
        
        A4.xx += best_shift[0]
        A4.xy += best_shift[1]
        A4.xz += best_shift[2] 
        current_dist = np.sqrt(distance2(A3, A4))
        A4.xx -= best_shift[0]
        A4.xy -= best_shift[1]
        A4.xz -= best_shift[2] 
        
        return check_dist, current_dist, best_shift
    
    def get_shift_from_rst(self, res1_number, atom1_name, res2_number, atom2_name, parm, new_rst):
        from pyxmol.geometry import distance, angle_deg, translate
        from parmed.amber import NetCDFTraj, Rst7, AmberParm 
        from parmed.geometry import distance2
        import numpy as np 

        for a in parm.atoms:
            if  a.residue.number == (res1_number-1) and a.name == atom1_name:
                A1 = a   
            elif a.residue.number == (res2_number-1) and a.name == atom2_name:
                A2 = a
        best_dist = np.sqrt(distance2(A1, A2))
        #self.log('start get_shift_from_rst')
        #self.log('best_dist: {}'.format(best_dist))
        A, B, C = self.make_current_lattice_vectors(new_rst.box) 
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    shift = i*A + j*B + k*C
                    #self.log('shift: {}'.format(shift))
                    A2.xx += shift[0]
                    A2.xy += shift[1]
                    A2.xz += shift[2] 
                    dist = np.sqrt(distance2(A1, A2))
                    #self.log('dist: {}, best_dist: {}, (best_dist-dist): {}'.format(round(dist, 9), round(best_dist, 9), best_dist-dist))
                    if round(dist, 9) <= round(best_dist, 9):
                        best_dist = dist
                        best_shift = shift 
                        #self.log('best shift: {}'.format(best_shift))
                    A2.xx -= shift[0]
                    A2.xy -= shift[1]
                    A2.xz -= shift[2] 
        return best_dist, best_shift
    
    def shift_rst(self, shift1, shift2, parm):
        from parmed.amber import AmberParm, Rst7, NetCDFTraj
        import os

        for r in parm.residues:
            if r.number > 56 and r.number <= 67:
                for a in r:
                    a.xx += shift1[0]
                    a.xy += shift1[1]
                    a.xz += shift1[2]
            if r.number ==68:
                for a in r:
                    a.xx += shift2[0]
                    a.xy += shift2[1]
                    a.xz += shift2[2]
            
        parm.write_rst7(MD._run_dir+"/run"+MD._pattern%self.current_step+"mod.rst")
        parm.save(MD._run_dir+"/run"+MD._pattern%self.current_step+"mod.pdb")
        self._restart_filename = MD._run_dir+"/run"+MD._pattern%self.current_step+"mod.rst"        
        
    def get_rst_from_nc(self, frame_number, shift, parm):
        from parmed.amber import AmberParm, Rst7, NetCDFTraj
        import os
        
        traj = NetCDFTraj.open_old(MD._run_dir+"/run"+MD._pattern%self.current_step+".nc")
        new_rst7 = Rst7(natom=traj.atom)
        new_rst7.coordinates = traj.coordinates[frame_number]
        new_rst7.vels = traj.velocities[frame_number]
        new_rst7.box = traj.box[frame_number]

        parm.load_rst7(new_rst7)
        for r in parm.residues:
            if r.number > 56 and r.number <= 67:
                for a in r:
                    a.xx += shift[0]
                    a.xy += shift[1]
                    a.xz += shift[2]
        parm.write_rst7(MD._run_dir+"/run"+MD._pattern%self.current_step+"mod.rst")
        parm.save(MD._run_dir+"/run"+MD._pattern%self.current_step+"mod.pdb")
        if not os.path.isdir(MD._run_dir+"/lock/"):
            os.mkdir(MD._run_dir+"/lock/")
        parm.save(MD._run_dir+"/lock/lock_begin.pdb")
        self._restart_filename = MD._run_dir+"/run"+MD._pattern%self.current_step+"mod.rst"
    
    
    
    def get_dist_and_angle(self, res1_number, atom1_name, res2_number, atom2_name, res3_number, atom3_name, parm):
        from parmed.amber import AmberParm, Rst7
        from parmed.geometry import distance2
        import numpy as np
        
        rst = Rst7(MD._run_dir+"/run"+MD._pattern%self.current_step+"mod.rst")
        parm.load_rst7(rst)
        for a in parm.atoms:
            if  a.residue.number == (res1_number-1) and a.name == atom1_name:
                A1 = a   
            elif a.residue.number == (res2_number-1) and a.name == atom2_name:
                A2 = a
            elif a.residue.number == (res3_number-1) and a.name == atom3_name:
                A3 = a
        A1_A2 = distance2(A1, A2)
        A1_A3 = distance2(A1, A3)
        A2_A3 = distance2(A2, A3)
        cos = (A1_A2 + A2_A3 - A1_A3)/(2*np.sqrt(A1_A2)*np.sqrt(A2_A3)) 
        A1_A2_A3 = np.degrees(np.arccos(cos)) 
        return np.sqrt(A1_A2),  A1_A2_A3      
    
        
    def get_lj_type(self, topology, selection, executable="parmed"):
                
        import subprocess
        import os

        assert os.path.isfile(topology)
        p = subprocess.Popen([executable,topology],stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        stdout, stderr = p.communicate("printLJTypes {}".format(selection))
        lines = stdout.split("\n")
        for i,l in enumerate(lines):
            if l.strip() == "---------------------------------------------":
                break

        i+=1

        assert len(lines)>i
        l = lines[i]

        result = None

        while (l.startswith("ATOM")):
            index_after = "Type index:"
            pos = l.find(index_after)
            assert (pos > 0)

            atom_type = int(l[pos+len(index_after):])
            if result is None:
                result = atom_type
            else:
                assert result == atom_type, "Selection `"+selection+"` matches different atom types (%d,%d)"%(result, atom_type)

            i+=1
            if len(lines)>i:
                l = lines[i]
            else:
                break

        assert result is not None

        return result

    def write_mod_topology(self, outname=None): # TODO FIXME
        import shutil
        import os

        if outname is None:
            outname = self.tleaprc.output_name + ".mod.prmtop"
        
        if len(self.parmed._commands)!=0:
            if os.path.isfile(outname+".bak"):
                os.remove(outname+".bak")
                
            self.call_cmd_pipe(
                ["parmed.py", self.tleaprc.output_name+".prmtop"],
                "\n".join(
                    self.parmed.get_commands()
                    +["outparm %s "%(outname+".bak")]
                    +["quit"]
                )
            )
            shutil.move(outname+".bak", outname)
        else:
            shutil.copyfile(self.tleaprc.output_name+".prmtop", outname)
    
            
    
    def gentle_lock(self, best_frame, lock_steps, lock_nstlim_per_step, shift):
        from pyxmol import aName, rName, aId, rId
        from pyxmol.geometry import distance,angle_deg
        from parmed.amber import AmberParm, Rst7, NetCDFTraj
        from pyxmol import readPdb
        from copy import deepcopy
        import os
        import shutil
        import subprocess
        import numpy as np
        import math
        
        #change md parameters
        self.log('lock_steps: {}'.format(lock_steps))
        nstlim_old = self.run_parameters["nstlim"]
        ntwr_old = self.run_parameters["ntwr"]
        ntwx_old=self.run_parameters["ntwx"]
        ntpr_old=self.run_parameters["ntpr"]
        self.run_parameters.set(nstlim=lock_nstlim_per_step, ntwr=lock_nstlim_per_step, ntwv=0, ntwx=lock_nstlim_per_step, ntpr=lock_nstlim_per_step)
        
        
        #change topology and Rst
        parm = AmberParm(MD._build_dir+"/box.prmtop")
        self.get_rst_from_nc(best_frame, shift, parm)
        current_dist, current_angle = self.get_dist_and_angle(32, "SG", 68, "C2", 68, "CL", parm)
        
        self.log("Bond dist before tying: %f" % (current_dist))
        self.log("Bond angle before tying: %f" % (current_angle))
         
        self.prepare_final_rst(parm)
        shutil.copyfile(MD._build_dir+"/box.prmtop", MD._build_dir+"/old_box.prmtop")
        shutil.copyfile(os.environ["PRMTOP_HOME"]+"/prmtop_linear/prmtops_new/box_before_lock_mod.prmtop", MD._build_dir+"/box.prmtop")
        shutil.copyfile(os.environ["PRMTOP_HOME"]+"/prmtop_linear/prmtops_new/box_before_lock_mod.prmtop", MD._build_dir+"/box.mod.prmtop")
        p = subprocess.Popen(["ambpdb", "-p",MD._build_dir+"/box.prmtop", "-c", self._restart_filename], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        stdout, stderr = p.communicate()
        with open(self.tleaprc.pdb_output_name+".pdb", "w") as fout:
            fout.write(stdout)
        self.restricted_structure = readPdb(self.tleaprc.pdb_output_name+".pdb")[0].asResidues >> (rId<=(self.n_residues+1))
        ats = self.restricted_structure.asAtoms
        
        
        #select atoms
        C2 = (ats >> ((aName == "C2") & (rName == "LYZ")))[0]
        C1 = (ats >> ((aName == "C1") & (rName == "LYZ")))[0]
        H1 = (ats >> ((aName == "H1") & (rName == "LYZ")))[0]
        H2 = (ats >> ((aName == "H2") & (rName == "LYZ")))[0]
        SG = (ats >> ((aName == "SG") & (rName == "CYZ")))[0]
        CB = (ats >> ((aName == "CB") & (rName == "CYZ")))[0]
        CL = (ats >> ((aName == "Cl-") & (rName == "Cl-")))[0]
              
        def lineal(x):
            return x
             
        self.log('Start linear transformation')
        t = 0
        while t < lock_steps:
            os.remove(MD._run_dir+"/run%05d.nc" % self.current_step)
            self.log('t: {}, lock_steps: {}'.format(t, lock_steps))
            x = float(t+1)/lock_steps

                
            self.log("manage C2-CL bond")
            self.run_parameters.add_distance_restraint(C2.aId, CL.aId, 0.0, 1.766, 1.766, 99.0, 232*(1-lineal(x)), 232*(1-lineal(x)), comment="")
            self.run_parameters.add_angle_restraint(CL.aId, C2.aId, C1.aId, 0.0, 110.41, 110.41, 180.0, 71.7*(1-lineal(x)), 71.7*(1-lineal(x)), comment="")
            self.run_parameters.add_angle_restraint(CL.aId, C2.aId, H1.aId, 0.0, 108.5, 108.5, 180.0, 50.0*(1-lineal(x)), 50.0*(1-lineal(x)), comment="")
            self.run_parameters.add_angle_restraint(CL.aId, C2.aId, H2.aId, 0.0, 108.5, 108.5, 180.0, 50.0*(1-lineal(x)), 50.0*(1-lineal(x)), comment="")
 

            self.log("manage C2-SG bond")
            self.run_parameters.add_distance_restraint(C2.aId, SG.aId, 0.0, 1.81, 1.81, 99.0, 227*lineal(x), 227*lineal(x), comment="")
            self.run_parameters.add_angle_restraint(SG.aId, C2.aId, C1.aId, 0.0, 108.84, 108.84, 180.0, 63.79*lineal(x), 63.79*lineal(x), comment="")
            self.run_parameters.add_angle_restraint(SG.aId, C2.aId, H1.aId, 0.0, 109.5, 109.5, 180.0, 50.0*lineal(x), 50.0*lineal(x), comment="")
            self.run_parameters.add_angle_restraint(SG.aId, C2.aId, H2.aId, 0.0, 109.5, 109.5, 180.0, 50.0*lineal(x), 50.0*lineal(x), comment="")        
            
     
            self.log("manage CB-SG bond")
            self.run_parameters.add_angle_restraint(CB.aId, SG.aId, C2.aId, 0.0, 98.9, 98.9, 180.0, 62*lineal(x), 62*lineal(x), comment="")
            
            shutil.copyfile(os.environ["PRMTOP_HOME"]+"/prmtop_linear/prmtops_new/box_{}_parmed.prmtop".format(t), MD._build_dir+"/box.prmtop")
            shutil.copyfile(os.environ["PRMTOP_HOME"]+"/prmtop_linear/prmtops_new/box_{}_parmed.prmtop".format(t), MD._build_dir+"/box.mod.prmtop")
            self.do_md_step()
            
            # delete current restraints
            self.run_parameters.del_distance_restraints(C2.aId, CL.aId)
            self.run_parameters.del_distance_restraints(C2.aId, SG.aId)
            self.run_parameters.del_angle_restraint(CL.aId, C2.aId, C1.aId)
            self.run_parameters.del_angle_restraint(CL.aId, C2.aId, H1.aId)
            self.run_parameters.del_angle_restraint(CL.aId, C2.aId, H2.aId)
            self.run_parameters.del_angle_restraint(SG.aId, C2.aId, C1.aId)
            self.run_parameters.del_angle_restraint(SG.aId, C2.aId, H1.aId)
            self.run_parameters.del_angle_restraint(SG.aId, C2.aId, H2.aId)            
            self.run_parameters.del_angle_restraint(CB.aId, SG.aId, C2.aId)
            
            #shift rst 
            new_rst = Rst7(MD._run_dir+"/run"+MD._pattern%self.current_step+".rst")
            parm = AmberParm(MD._build_dir+"/box.mod.prmtop")
            parm.load_rst7(new_rst)
            dist1, shift1 = self.get_shift_from_rst(32, "SG", 68, "C2", parm, new_rst)
            #self.log("Shift peptide: ax = %f, ay = %f, az = %f" % (shift1[0], shift1[1], shift1[2]), MD.INFO)
            dist2, shift2 = self.get_shift_from_rst(32, "SG", 69, "Cl-", parm, new_rst)   
            #self.log("Shift Cl-: ax = %f, ay = %f, az = %f" % (shift2[0], shift2[1], shift2[2]), MD.INFO)
            self.shift_rst(shift1, shift2, parm)
             
            
            self.put_frame(ats, -1, -1)
            ats.writeAsPdb(open(MD._run_dir+"/run%05d.pdb" % self.current_step, "w"))
            if not os.path.isdir(MD._run_dir+"/lock/"):
                os.mkdir(MD._run_dir+"/lock/")
            ats.writeAsPdb(open(MD._run_dir+"/lock/"+"lock-%d.pdb" % t, "w"))
            t += 1
        self.log('Finish linear transformation')    

                
        self.run_parameters.set(nstlim= nstlim_old, ntwr=ntwr_old, ntwx=ntwx_old, ntpr=ntpr_old)
        self.log('set old nstlim: {}'.format(nstlim_old))
        self.log('set old ntwr: {}'.format(ntwr_old))
        self.log('set old ntwx: {}'.format(ntwx_old))
        self.log('set old ntpr: {}'.format(ntpr_old))
        self.set_step_as_restart_file()
        self._restart_filename = MD._run_dir+"/run"+MD._pattern%self.current_step+"mod.rst"
    
    
    def run_continue(self):
        import random
        from pyxmol import aName, rName, aId, rId
        from pyxmol.geometry import distance, angle_deg
        from pyxmol import readPdb
        import shutil
        import subprocess
        import numpy as np
        import os
        from parmed.amber import AmberParm, Rst7, NetCDFTraj
        #self.keep_netcdf = False
        
        self.log("Continue > ")

        self.log("")
        self.log("Unconstrained simulation")
        self.log("")
        
        ats = self.restricted_structure.asAtoms
        bond = None
        step_limit = 5000
        self.log("TRAJECTORY_PART_01_FIRST_STEP={}".format(1))
        while self.current_step < 500:
            self.do_md_step()
            self.put_frame(ats, -1, -1)
            ats.writeAsPdb(open(MD._run_dir+"/run%05d.pdb" % self.current_step, "w"))
            os.remove(MD._run_dir+"/run%05d.nc" % self.current_step)
        
        self.run_parameters["ntwv"] = -1
        parm = AmberParm(MD._build_dir+"/box.prmtop")
        while self.current_step < step_limit:  
            self.do_md_step()
            self.put_frame(ats, -1, -1)
            ats.writeAsPdb(open(MD._run_dir+"/run%05d.pdb" % self.current_step, "w"))
            if bond is None:
                best_frame = None
                best_distance = 40
                SG = (ats >> ((aName == "SG") & (rName == "CYM")))[0]
                C2 = (ats >> ((aName == "C2") & (rName == "LYC")))[0]
                CL = (ats >> ((aName == "CL") & (rName == "LYC")))[0]
                O = (ats >> ((aName == "O") & (rName == "PRO") & (rId == 64)))[0]
                N = (ats >> ((aName == "NE1") & (rName == "TRP") & (rId == 36)))[0]
                traj = NetCDFTraj.open_old(MD._run_dir+"/run"+MD._pattern%self.current_step+".nc")
                
                for frame in range(1000):
                    rand = 10
                    self.put_frame(ats, -1, frame)
                    check_dist = distance(N, O)
                    current_dist = distance(SG, C2)
                    shift = np.array([0, 0, 0])
                    self.log("Check distance: dist = %f" % (check_dist), MD.INFO)
                    self.log("SG-C2 distance: dist = %f" % (current_dist), MD.INFO)
                    if check_dist > 5:
                        check_dist, current_dist, shift = self.get_shift_from_nc_2(frame, 36, "NE1", 64, "O", parm, traj, 32, "SG", 68, "C2")
                        self.log("Corrected check distance: dist = %f" % (check_dist), MD.INFO)
                        self.log("Shift: ax = %f, ay = %f, az = %f" % (shift[0], shift[1], shift[2]), MD.INFO)
                        self.log("Corrected SC-C2 distance: dist = %f" % (current_dist), MD.INFO)
                    if current_dist < 3.3:
                        self.log("Throw the dice...")
                        rand = random.random()
                    if rand < 0.1:
                        self.log("You win. Random is {} ".format(rand), MD.INFO) 
                        best_distance = current_dist
                        best_frame = frame
                        best_shift = shift
                        break
                   
                if best_frame is not None:
                    bond = SG, C2
                    self.log("best_frame = {}".format(best_frame), MD.INFO)
                    self.log("Bond found: dist = %f" % (best_distance), MD.INFO)
                else:
                    self.log("Bond isn't found")

                if bond is not None:
                    self.log("Start gentle lock...")
                    self.log("TRAJECTORY_PART_01_LAST_STEP={}".format(self.current_step-1))
                    self.log("TRAJECTORY_PART_02_FIRST_STEP={}".format(self.current_step+1))
                    self.write_mod_topology(outname="1_build/box_before_lock.prmtop")               
                    lock_steps=200
                    lock_nstlim_per_step=5000
                    self.gentle_lock(best_frame, lock_steps, lock_nstlim_per_step, best_shift)               
                    self.log("Finish gentle lock.")
                    shutil.copyfile(MD._build_dir+"/final_box.mod.prmtop", MD._build_dir+"/box.mod.prmtop")
                    p = subprocess.Popen(["ambpdb", "-p",MD._build_dir+"/box.mod.prmtop", "-c", self._restart_filename], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                    stdout, stderr = p.communicate()
                    with open(self.tleaprc.pdb_output_name+".pdb", "w") as fout:
                        fout.write(stdout)
                    self.restricted_structure = readPdb(self.tleaprc.pdb_output_name+".pdb")[0].asResidues >> (rId<=(self.n_residues+1))
                    ats = self.restricted_structure.asAtoms
                    self.log("TRAJECTORY_PART_02_LAST_STEP={}".format(self.current_step))
                    self.log("TRAJECTORY_PART_03_FIRST_STEP={}".format(self.current_step+1))
                    
                    step_limit = self.current_step + 500;
            os.remove(MD._run_dir+"/run%05d.nc" % self.current_step)
        
        while self.current_step < 500:
            self.run_parameters["ig"] = random.randint(1,100000)
            self.log("Random seed is %d " % self.run_parameters["ig"])
            self.do_md_step()
            self.put_frame(ats, -1, -1)
            ats.writeAsPdb(open(MD._run_dir+"/run%05d.pdb" % self.current_step, "w"))
            os.remove(MD._run_dir+"/run%05d.nc" % self.current_step)
       
        self.log("TRAJECTORY_PART_03_LAST_STEP={}".format(self.current_step))
        self.save_state("all_done.pickle")

        self.log("Continue < ")


import os
import glob
wd = os.path.abspath(".")
for structs in [("prepare/start.pdb", "prepare/final.pdb")]:
    os.chdir(wd)
    MD_grb2(*structs)



