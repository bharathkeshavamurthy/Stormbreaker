What is this?
=============

This patch set introduced another compile-time and build-time dependency for
BAM!  Radio: [Embeddable Common Lisp v16.1.3][1]. There is a release tarball
available to download [here][2]. You will also find this tarball on bam-build
in the /repos folder.

As a result of this extra dependency, bam-base was bumped to version number
0.5. The bare container on the colosseum was updated to include this
dependency.

To install this software in your dev environment, I recommend building it from
source:

```shell
... untar, cd, etc...
./configure --enable-shared --enable-threads --enable-boehm=included --with-dffi
make # no parallel make here
sudo make install && sudo ldconfig
```

Note that Ubuntu Xenial does include an ECL package, but it seems like it is
only version 15.xx, which is not compatible with our code.

This extra dependency introduces a substantial new feature: We can now write
code for BAM! Radio using ANSI Common Lisp (CL), a dynamic, high-level language
that has been a staple in the AI community for many decades. This enables us to
include libraries from the [rich CL ecosystem][3] and to program our algorithms
with a language better suited for such a high-level job.

Examples of use
---------------

This commit only scratches the surface of the integration. For now, check out
relevant sections in de.cc for an "Hello, World!"-style example: in the
`_step(...)` routine, we now run LISP code called `decision-engine-step`, which
simply echoes the trigger type using our logging framework. I expect things to
naturally evolve from there. I have somewhat vague plans of using a big "input
object" (this is already somewhat outlined in `data.lisp`) to move data "into
LISP", then do some computations to make decisions, and then call some
DecisionEngine methods to execute them using the `LISPY_METHOD(...)` trick (see
`lisp.h` for this mechanism.)

A Note on CL dependencies
-------------------------

It is expected that we make heavy use of external libraries in our LISP
code. There are some nifty AI libraries that seem very useful. I recommend
using [Quicklisp][3] to pull them on to your workstation during development. To
build a reproducible bam-radio release, I mirror all libraries that are needed
to build and run the radio to our cloudradio redmine repository. I furthermore
maintain a meta-repository called "lisp-deps" that includes all dependencies as
submodules. This creates a multi-layered git submodule tree, so you need to
make sure to always recursively checkout the repository. To pull in updates to
all submodules, we can do

```shell
git submodule update --init --recursive
```

Also see how we do this in util/make_release.sh. Anyways, the easiest option
when you want to include a specific lib is to just have me add it to the
lisp-deps submodule.

A Note on picking up the language
---------------------------------

A good place to start is [this blog post][4]. To set up a dev environment, I
recommend [SLIME][5], but there are vim alternatives described in [4]. Working
through some chapters [Practical Common Lisp][6] is a good idea to get a
feel. As references, it seems that [the cookbook][7] is human-friendly and
you'll probably find yourself browsing [the spec][8] quite a bit.

A lot of places implicitly assume that you're working with SBCL or GNU Common
Lisp, or something along those lines. I don't expect anything that we might be
doing to *not* work on ECL. I am using the ECL implementaion during development
as my REPL and it seems to work well. You start it using the `ecl`
command. YMMV, but SLIME should just work.

You can find the [ECL docs here][9], but I must warn you that they do not seem
to be in sync with v16.1.3, so be careful. I found that it is helpful to read
the ECL source code together with the docs to settle any discrepancies. To get
a little more of a grip on the mechanics of ECL, I found [this tutorial][10] to
be a helpful read. I have taken care of most of this stuff in the files
`lisp.h` and `lisp.cc`. You can find examples on how to convert to and from
LISP objects in `de.h`.

References
----------

[1]  https://common-lisp.net/project/ecl
[2]  https://common-lisp.net/project/ecl/static/files/release/ecl-16.1.3.tgz
[3]  https://www.quicklisp.org/beta/
[4]  http://stevelosh.com/blog/2018/08/a-road-to-common-lisp/
[5]  https://common-lisp.net/project/slime/
[6]  http://www.gigamonkeys.com/book/
[7]  https://lispcookbook.github.io/cl-cookbook/
[8]  http://www.lispworks.com/documentation/HyperSpec/Front/
[9]  https://common-lisp.net/project/ecl/static/ecldoc/
[10] https://common-lisp.net/project/ecl/index.html#orgheadline10
