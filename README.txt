paper-phonograph
================

The code in this repository relies on data stored elsewhere. In order to allow
people to store that data anywhere on their machine, this code expects that it
is located in the location specified by the environment variable
PAPER_PHONOGRAPH_DROPBOX.

On UNIX systems, define this environment variable by adding::

    export PAPER_PHONOGRAPH_DROPBOX=/path/to/Dropbox/paper-phonograph/

to your `~/.bashrc` file. For example, on my system, I have something like
/home/fitze/Dropbox/paper-phonograph. For Windows, see `setx`
(http://superuser.com/questions/79612/setting-and-getting-windows-environment-variables-from-the-command-prompt)
or http://support.microsoft.com/kb/310519.

You can also set the environment variable temporarily in your session of python
via::

   `>>> import os; os.environ['PAPER_PHONOGRAPH_DROPBOX'] = '/path/to/'
