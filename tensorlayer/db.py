#! /usr/bin/python
# -*- coding: utf8 -*-
"""
Experimental Database Management System.

Latest Version
"""
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import time
import math
from lz4.frame import compress, decompress

import uuid

import pymongo
import gridfs
import pickle
from pymongo import MongoClient
from datetime import datetime

import inspect


class JobStatus(object):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    TERMINATED = "terminated"


def AutoFill(func):
    def func_wrapper(self,*args,**kwargs):
        d=inspect.getcallargs(func,self,*args,**kwargs)
        d['args'].update({"studyID":self.studyID})
        return  func(**d)
    return func_wrapper


class TensorDB(object):
    """TensorDB is a MongoDB based manager that help you to manage data, network topology, parameters and logging.

    Parameters
    -------------
    ip : string, localhost or IP address.
    port : int, port number.
    db_name : string, database name.
    user_name : string, set to None if it donnot need authentication.
    password : string.

    Properties
    ------------
    db : ``pymongo.MongoClient[db_name]``, xxxxxx
    datafs : ``gridfs.GridFS(self.db, collection="datafs")``, xxxxxxxxxx
    modelfs : ``gridfs.GridFS(self.db, collection="modelfs")``,
    paramsfs : ``gridfs.GridFS(self.db, collection="paramsfs")``,
    db.Params : Collection for
    db.TrainLog : Collection for
    db.ValidLog : Collection for
    db.TestLog : Collection for
    studyID : string or None, the unique study ID, if None random generate one.

    Dependencies
    -------------
    1 : MongoDB, as TensorDB is based on MongoDB, you need to install it in your
       local machine or remote machine.
    2 : pip install pymongo, for MongoDB python API.

    Optional Tools
    ----------------
    1 : You may like to install MongoChef or Mongo Management Studo APP for
       visualizing or testing your MongoDB.
    """
    def __init__(
        self,
        ip = 'localhost',
        port = 27017,
        db_name = 'db_name',
        user_name = None,
        password = 'password',
        studyID = None
    ):
        ## connect mongodb
        client = MongoClient(ip, port)
        self.db = client[db_name]
        if user_name != None:
            self.db.authenticate(user_name, password)

        if studyID is None:
            self.studyID=str(uuid.uuid1())
        else:
            self.studyID=studyID

        ## define file system (Buckets)
        self.datafs = gridfs.GridFS(self.db, collection="datafs")
        self.modelfs = gridfs.GridFS(self.db, collection="modelfs")
        self.paramsfs = gridfs.GridFS(self.db, collection="paramsfs")
        self.archfs=gridfs.GridFS(self.db,collection="ModelArchitecture")
        ##
        print(("[TensorDB] Connect SUCCESS {}:{} {} {} {}".format(ip, port, db_name, user_name, studyID)))

        self.ip = ip
        self.port = port
        self.db_name = db_name
        self.user_name = user_name

    def __autofill(self,args):
        return args.update({'studyID':self.studyID})

    def __serialization(self,ps):
        return pickle.dumps(ps, protocol=2)

    def __deserialization(self,ps):
        return pickle.loads(ps)

    def save_params(self, params=[], args={}, lz4_comp=False):#, file_name='parameters'):
        """ Save parameters into MongoDB Buckets, and save the file ID into Params Collections.

        Parameters
        ----------
        params : a list of parameters
        args : dictionary, item meta data.

        Returns
        ---------
        f_id : the Buckets ID of the parameters.
        """
        self.__autofill(args)
        st = time.time()
        d = self.__serialization(params)
        # print('seri time', time.time()-st)

        if lz4_comp:
            # s = time.time()
            d = compress(d, compression_level=3)
            # print('comp time', time.time()-s)

        # s = time.time()
        f_id = self.paramsfs.put(d)#, file_name=file_name)
        # print('save time', time.time()-s)
        args.update({'f_id': f_id, 'time': datetime.utcnow()})
        self.db.Params.insert_one(args)
        # print("[TensorDB] Save params: {} SUCCESS, took: {}s".format(file_name, round(time.time()-s, 2)))
        print(("[TensorDB] Save params: SUCCESS, took: {}s".format(round(time.time()-st, 2))))
        return f_id

    @AutoFill
    def find_one_params(self, args={}, sort=None, lz4_decomp=False):
        """ Find one parameter from MongoDB Buckets.

        Parameters
        ----------
        args : dictionary, find items.

        Returns
        --------
        params : the parameters, return False if nothing found.
        f_id : the Buckets ID of the parameters, return False if nothing found.
        """
        d = self.db.Params.find_one(filter=args, sort=sort)

        if d is not None:
            f_id = d['f_id']
        else:
            print(("[TensorDB] Cannot find: {}".format(args)))
            return False, False

        st = time.time()
        d = self.paramsfs.get(f_id).read()
        # print('get time', time.time()-st)

        if lz4_decomp:
            # s = time.time()
            d = decompress(d)
            # print('decomp time', time.time()-s)

        # s = time.time()
        params = self.__deserialization(d)
        # print('deseri time', time.time()-s)

        print(("[TensorDB] Find one params SUCCESS, {} took: {}s".format(args, round(time.time()-st, 2))))
        return params, f_id

    @AutoFill
    def find_all_params(self, args={}, lz4_decomp=False):
        """ Find all parameter from MongoDB Buckets

        Parameters
        ----------
        args : dictionary, find items

        Returns
        --------
        params : the parameters, return False if nothing found.

        """
        st = time.time()
        pc = self.db.Params.find(args)

        if pc is not None:
            f_id_list = pc.distinct('f_id')
            params = []
            for f_id in f_id_list: # you may have multiple Buckets files
                tmp = self.paramsfs.get(f_id).read()
                if lz4_decomp:
                    tmp = decompress(tmp)
                params.append(self.__deserialization(tmp))
        else:
            print(("[TensorDB] Cannot find: {}".format(args)))
            return False

        print(("[TensorDB] Find all params SUCCESS, took: {}s".format(round(time.time()-st, 2))))
        return params

    @AutoFill
    def del_params(self, args={}):
        """ Delete params in MongoDB uckets.

        Parameters
        -----------
        args : dictionary, find items to delete, leave it empty to delete all parameters.
        """
        pc = self.db.Params.find(args)
        f_id_list = pc.distinct('f_id')
        # remove from Buckets
        for f in f_id_list:
            self.paramsfs.delete(f)
        # remove from Collections
        self.db.Params.remove(args)

        print(("[TensorDB] Delete params SUCCESS: {}".format(args)))

    def _print_dict(self, args):
        """ """
        # return " / ".join(str(key) + ": "+ str(value) for key, value in args.items())
        string = ''
        for key, value in list(args.items()):
            if key is not '_id':
                string += str(key) + ": "+ str(value) + " / "
        return string

    ## =========================== LOG =================================== ##
    @AutoFill
    def train_log(self, args={}):
        """Save the training log.

        Parameters
        -----------
        args : dictionary, items to save.

        Examples
        ---------
        >>> db.train_log(time=time.time(), {'loss': loss, 'acc': acc})
        """
        _result = self.db.TrainLog.insert_one(args)
        _log = self._print_dict(args)
        #print("[TensorDB] TrainLog: " +_log)
        return _result

    @AutoFill
    def del_train_log(self, args={}):
        """ Delete train log.

        Parameters
        -----------
        args : dictionary, find items to delete, leave it empty to delete all log.
        """

        self.db.TrainLog.delete_many(args)
        print("[TensorDB] Delete TrainLog SUCCESS")

    @AutoFill
    def valid_log(self, args={}):
        """Save the validating log.

        Parameters
        -----------
        args : dictionary, items to save.

        Examples
        ---------
        >>> db.valid_log(time=time.time(), {'loss': loss, 'acc': acc})
        """

        _result = self.db.ValidLog.insert_one(args)
        # _log = "".join(str(key) + ": " + str(value) for key, value in args.items())
        _log = self._print_dict(args)
        print(("[TensorDB] ValidLog: " +_log))
        return _result

    @AutoFill
    def del_valid_log(self, args={}):
        """ Delete validation log.

        Parameters
        -----------
        args : dictionary, find items to delete, leave it empty to delete all log.
        """
        self.db.ValidLog.delete_many(args)
        print("[TensorDB] Delete ValidLog SUCCESS")

    @AutoFill
    def test_log(self, args={}):
        """Save the testing log.

        Parameters
        -----------
        args : dictionary, items to save.

        Examples
        ---------
        >>> db.test_log(time=time.time(), {'loss': loss, 'acc': acc})
        """

        _result = self.db.TestLog.insert_one(args)
        # _log = "".join(str(key) + str(value) for key, value in args.items())
        _log = self._print_dict(args)
        print(("[TensorDB] TestLog: " +_log))
        return _result

    @AutoFill
    def del_test_log(self, args={}):
        """Delete test log.

        Parameters
        -----------
        args : dictionary, find items to delete, leave it empty to delete all log.
        """

        self.db.TestLog.delete_many(args)
        print("[TensorDB] Delete TestLog SUCCESS")

    ## =========================== Network Architecture ================== ##
    @AutoFill
    def save_model_architecture(self,s,args={}):
        """ """
        self.__autofill(args)
        fid=self.archfs.put(s,filename="modelarchitecture")
        args.update({"fid":fid})
        self.db.march.insert_one(args)

    @AutoFill
    def load_model_architecture(self,args={}):
        """ """
        d = self.db.march.find_one(args)
        if d is not None:
            fid = d['fid']
            print(d)
            print(fid)
            # "print find"
        else:
            print(("[TensorDB] Cannot find: {}".format(args)))
            print ("no idtem")
            return False, False
        try:
            archs = self.archfs.get(fid).read()
            '''print("[TensorDB] Find one params SUCCESS, {} took: {}s".format(args, round(time.time()-s, 2)))'''
            return archs, fid
        except Exception as e:
            print("exception")
            print(e)
            return False, False

    @AutoFill
    def submit_job(self, args={}, allow_duplicate=True):
        """ Submit a job.

        Parameters
        -----------
        args : dictionary, arguments of each job.
        allow_duplicate : bool, allow to submit the same job with the same arguments.

        Examples
        ---------
        >>> result = db.submit_job(args={
        ...     "file": "main.py",
        ...     "args": "--data_dir=/data",
        ... }, allow_duplicate=True)
        """
        self.__autofill(args)
        args.update({
            'status': JobStatus.WAITING,
            "datetime": datetime.utcnow(),
        })
        if allow_duplicate:
            _result = self.db.Job.insert_one(args)
        else:
            _result = self.db.Job.replace_one(args, args, upsert=True)
        _log = self._print_dict(args)
        print(("[TensorDB] Submit Job: args={}".format(args)))
        return _result

    def get_job(self, job_id):
        """ Get a job by ID.

        Parameters
        -----------
        job_id : ObjectId, job id from MongoDB.

        Examples
        ---------
        - Manually specify job id
        >>> from bson.objectid import ObjectId
        >>> job = db.get_job(job_id=ObjectId('5929da7f130fd737204369b3'))
        """
        job = self.db.Job.find_one({"_id": job_id})

        if job is None:
            print(("[TensorDB] Cannot find any job with id: {}".format(job_id)))
            return False
        else:
            return job

    def get_jobs(self, status=None):
        """ Get jobs based on their status.

        Parameters
        -----------
        status : string, status of jobs from tl.db.JobStatus

        Examples
        ---------
        - Get all running jobs
        >>> jobs = db.get_jobs(status=JobStatus.RUNNING)
        """
        jobs = []

        if status is None:
            cursor = self.db.Job.find({})
        else:
            cursor = self.db.Job.find({'status': status})

        if cursor is not None:
            for job in cursor: # you may have multiple Buckets files
                jobs.append(job)
        else:
            print("[TensorDB] There is no job")
            return False

        print(("[TensorDB] Get jobs with status:{} SUCCESS".format(status)))
        return jobs

    def change_job_status(self, job_id, status):
        """ Change the status of a job.

        Parameters
        -----------
        job_id : ObjectId, job id from MongoDB.
        status : string, status of jobs from tl.db.JobStatus

        Examples
        ---------
        - Terminate running jobs
        >>> jobs = db.get_jobs(status=JobStatus.RUNNING)
        >>> for j in jobs:
        >>>     print db.change_job_status(job_id=j["_id"], status=JobStatus.TERMINATED)
        """
        job = self.db.Job.find_one({"_id": job_id})

        if job is None:
            print(("[TensorDB] Cannot find any job with id: {}".format(job_id)))
            return False
        else:
            _result = self.db.Job.update(
                {'_id': job_id},
                {
                    '$set': {
                        'status': status, 
                        "datetime": datetime.utcnow()
                    }
                }, upsert=False, multi=False
            )
            print(("[TensorDB] Change the status of job ({}) to: {}".format(job_id, status)))
            return _result

    def __str__(self):
        _s = "[TensorDB] Info:\n"
        _t = _s + "    " + str(self.db)
        return _t

    # def save_bulk_data(self, data=None, filename='filename'):
    #     """ Put bulk data into TensorDB.datafs, return file ID.
    #     When you have a very large data, you may like to save it into GridFS Buckets
    #     instead of Collections, then when you want to load it, XXXX
    #
    #     Parameters
    #     -----------
    #     data : serialized data.
    #     filename : string, GridFS Buckets.
    #
    #     References
    #     -----------
    #     - MongoDB find, xxxxx
    #     """
    #     s = time.time()
    #     f_id = self.datafs.put(data, filename=filename)
    #     print("[TensorDB] save_bulk_data: {} took: {}s".format(filename, round(time.time()-s, 2)))
    #     return f_id
    #
    # def save_collection(self, data=None, collect_name='collect_name'):
    #     """ Insert data into MongoDB Collections, return xx.
    #
    #     Parameters
    #     -----------
    #     data : serialized data.
    #     collect_name : string, MongoDB collection name.
    #
    #     References
    #     -----------
    #     - MongoDB find, xxxxx
    #     """
    #     s = time.time()
    #     rl = self.db[collect_name].insert_many(data)
    #     print("[TensorDB] save_collection: {} took: {}s".format(collect_name, round(time.time()-s, 2)))
    #     return rl
    #
    # def find(self, args={}, collect_name='collect_name'):
    #     """ Find data from MongoDB Collections.
    #
    #     Parameters
    #     -----------
    #     args : dictionary, arguments for finding.
    #     collect_name : string, MongoDB collection name.
    #
    #     References
    #     -----------
    #     - MongoDB find, xxxxx
    #     """
    #     s = time.time()
    #
    #     pc = self.db[collect_name].find(args)  # pymongo.cursor.Cursor object
    #     flist = pc.distinct('f_id')
    #     fldict = {}
    #     for f in flist: # you may have multiple Buckets files
    #         # fldict[f] = pickle.loads(self.datafs.get(f).read())
    #         # s2 = time.time()
    #         tmp = self.datafs.get(f).read()
    #         # print(time.time()-s2)
    #         fldict[f] = pickle.loads(tmp)
    #         # print(time.time()-s2)
    #         # exit()
    #     # print(round(time.time()-s, 2))
    #     data = [fldict[x['f_id']][x['id']] for x in pc]
    #     data = np.asarray(data)
    #     print("[TensorDB] find: {} get: {} took: {}s".format(collect_name, pc.count(), round(time.time()-s, 2)))
    #     return data



class DBLogger:
    """ """
    def __init__(self,db,model):
        self.db=db
        self.model=model

    def on_train_begin(self,logs={}):
        print("start")

    def on_train_end(self,logs={}):
        print("end")

    def on_epoch_begin(self,epoch,logs={}):
        self.epoch=epoch
        self.et=time.time()
        return

    def on_epoch_end(self, epoch, logs={}):
        self.et=time.time()-self.et
        print("ending")
        print(epoch)
        logs['epoch']=epoch
        logs['time']=datetime.utcnow()
        logs['stepTime']=self.et
        logs['acc']=np.asscalar(logs['acc'])
        print(logs)

        w=self.model.Params
        fid=self.db.save_params(w,logs)
        logs.update({'params':fid})
        self.db.valid_log(logs)
    def on_batch_begin(self, batch,logs={}):
        self.t=time.time()
        self.losses = []
        self.batch=batch

    def on_batch_end(self, batch, logs={}):
        self.t2=time.time()-self.t
        logs['acc']=np.asscalar(logs['acc'])
        #logs['loss']=np.asscalar(logs['loss'])
        logs['step_time']=self.t2
        logs['time']=datetime.utcnow()
        logs['epoch']=self.epoch
        logs['batch']=self.batch
        self.db.train_log(logs)
