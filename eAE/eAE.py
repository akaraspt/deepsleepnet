#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Provides methods for accessing the interface eAE API.

"""
import json
import stat
import os
import zipfile
import http.client as http

from subprocess import call
from uuid import uuid4

__author__ = "Axel Oehmichen"
__copyright__ = "Copyright 2017, Axel Oehmichen"
__credits__ = []
__license__ = "Apache 2"
__version__ = "0.1"
__maintainer__ = "Axel Oehmichen"
__email__ = "ao1011@imperial.ac.uk"
__status__ = "Dev"

__all__ = ['eAE']


class eAE(object):

    def __init__(self, interface_ip, interface_port):
        self.interface_ip = str(interface_ip)
        self.interface_port = int(interface_port)
        self.connection = http.HTTPSConnection(self.interface_ip, self.interface_port)

    def __str__(self):
        return "\rThe interface ip is set to: {0}\r The interface port is set to: {1}".format(self.interface_ip,
                                                                                            self.interface_port)

    def _create_eae_zipfile(self, zip_file_name, main_file_path, data_files=None):

        to_zip = []
        if data_files is None:
            data_files = []

        # Handle main script
        to_zip.append(main_file_path)

        # Prepare the zip file
        zip_path = "/tmp/" + zip_file_name
        zipf = zipfile.ZipFile(zip_path, mode='w', compression=zipfile.ZIP_DEFLATED, allowZip64=True)
        for f in to_zip:
            zipf.write(f)
        zipf.close()

        # Handle other files & dirs
        for f in data_files:
            zipCommand = "zip -r -u -0 " + zip_path + " " + f
            call([zipCommand], shell=True)

        # Chmod 666 the zip file so it can be accessed
        os.chmod(zip_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)

        return zip_path

    def is_eae_alive(self):
        """Retrieve the status of the eAE"""
        self.connection.request('GET', '/interfaceEAE/utilities/isAlive')
        res = self.connection.getresponse()
        return int(res.read())

    def retrieve_clusters(self):
        """Retrieve the list of all available clusters"""
        self.connection.request('GET', '/interfaceEAE/EAEManagement/retrieveClusters')
        res = self.connection.getresponse()
        str_response = res.read().decode('utf-8')
        clusters = json.loads(str_response)
        return clusters

    def submit_jobs(self, parameters_set, cluster, computation_type, main_file, data_files, host_ip, ssh_port="22"):
        """Submit jobs to the eAE backend
        
        This method is called when a specific task needs to be deployed on a cluster.
        """

        uuid = uuid4()
        zip_file_name = "{0}.zip".format(uuid)
        configs = parameters_set
        zip_file = self._create_eae_zipfile(zip_file_name, main_file, data_files)
        data = {'id': str(uuid), 'host_ip': host_ip, 'ssh_port': ssh_port, 'zip': zip_file, 'configs': configs,
                'cluster': cluster, 'clusterType': computation_type, 'mainScriptExport': main_file}
        data_str = json.dumps(data)
        self.connection.request('POST', '/interfaceEAE/OpenLava/submitJob', data_str)
        res = self.connection.getresponse()
        submit_sucess = res.read()

        return submit_sucess


def test_methods():
    # Setting up the connection to interface
    ip = "interfaceeae.doc.ic.ac.uk"
    port = 443
    eae = eAE(ip, port)

    # Testing if the interface is Alive
    is_alive = eae.is_eae_alive()
    print(is_alive)

    # We retrieve the list of Clusters
    clusters = eae.retrieve_clusters()
    print(clusters)

    # We submit a dummy job
    parameters_set = "0\n 1\n 2\n"
    cluster = "python_large"
    computation_type = "Python"
    main_file = "/PATH/TO/FILE/Demo.py"
    data_files = ['']
    host_ip = "X.X.X.X"
    ssh_port = "22"
    job = eae.submit_jobs(parameters_set, cluster, computation_type, main_file, data_files, host_ip, ssh_port)
    print(job)

if __name__ == '__main__':
    test_methods()

