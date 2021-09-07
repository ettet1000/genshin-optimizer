(this["webpackJsonpgenshin-optimizer"]=this["webpackJsonpgenshin-optimizer"]||[]).push([[20],{532:function(e,t,n){"use strict";var a=n(137),c=n(0),r=n.n(c),i=n(36),o=n(131),s=n(29),l=function(e){var t=Object(i.a)(e,{activeKey:"onSelect"}),n=t.id,a=t.generateChildId,l=t.onSelect,d=t.activeKey,b=t.transition,j=t.mountOnEnter,u=t.unmountOnExit,h=t.children,x=Object(c.useMemo)((function(){return a||function(e,t){return n?n+"-"+t+"-"+e:null}}),[n,a]),O=Object(c.useMemo)((function(){return{onSelect:l,activeKey:d,transition:b,mountOnEnter:j||!1,unmountOnExit:u||!1,getControlledId:function(e){return x(e,"tabpane")},getControllerId:function(e){return x(e,"tab")}}}),[l,d,b,j,u,x]);return r.a.createElement(o.a.Provider,{value:O},r.a.createElement(s.a.Provider,{value:l||null},h))},d=n(2),b=n(5),j=n(6),u=n.n(j),h=n(8),x=["bsPrefix","as","className"],O=r.a.forwardRef((function(e,t){var n=e.bsPrefix,a=e.as,c=void 0===a?"div":a,i=e.className,o=Object(b.a)(e,x),s=Object(h.a)(n,"tab-content");return r.a.createElement(c,Object(d.a)({ref:t},o,{className:u()(i,s)}))})),m=n(60),f=["activeKey","getControlledId","getControllerId"],p=["bsPrefix","className","active","onEnter","onEntering","onEntered","onExit","onExiting","onExited","mountOnEnter","unmountOnExit","transition","as","eventKey"];var v=r.a.forwardRef((function(e,t){var n=function(e){var t=Object(c.useContext)(o.a);if(!t)return e;var n=t.activeKey,a=t.getControlledId,r=t.getControllerId,i=Object(b.a)(t,f),l=!1!==e.transition&&!1!==i.transition,j=Object(s.b)(e.eventKey);return Object(d.a)({},e,{active:null==e.active&&null!=j?Object(s.b)(n)===j:e.active,id:a(e.eventKey),"aria-labelledby":r(e.eventKey),transition:l&&(e.transition||i.transition||m.a),mountOnEnter:null!=e.mountOnEnter?e.mountOnEnter:i.mountOnEnter,unmountOnExit:null!=e.unmountOnExit?e.unmountOnExit:i.unmountOnExit})}(e),a=n.bsPrefix,i=n.className,l=n.active,j=n.onEnter,x=n.onEntering,O=n.onEntered,v=n.onExit,g=n.onExiting,y=n.onExited,E=n.mountOnEnter,K=n.unmountOnExit,C=n.transition,_=n.as,w=void 0===_?"div":_,S=(n.eventKey,Object(b.a)(n,p)),k=Object(h.a)(a,"tab-pane");if(!l&&!C&&K)return null;var N=r.a.createElement(w,Object(d.a)({},S,{ref:t,role:"tabpanel","aria-hidden":!l,className:u()(i,k,{active:l})}));return C&&(N=r.a.createElement(C,{in:l,onEnter:j,onEntering:x,onEntered:O,onExit:v,onExiting:g,onExited:y,mountOnEnter:E,unmountOnExit:K},N)),r.a.createElement(o.a.Provider,{value:null},r.a.createElement(s.a.Provider,{value:null},N))}));v.displayName="TabPane";var g=v,y=function(e){function t(){return e.apply(this,arguments)||this}return Object(a.a)(t,e),t.prototype.render=function(){throw new Error("ReactBootstrap: The `Tab` component is not meant to be rendered! It's an abstract component that is only valid as a direct Child of the `Tabs` Component. For custom tabs components use TabPane and TabsContainer directly")},t}(r.a.Component);y.Container=l,y.Content=O,y.Pane=g;t.a=y},533:function(e,t,n){"use strict";var a=n(2),c=n(5),r=n(6),i=n.n(r),o=n(0),s=n.n(o),l=(n(53),n(36)),d=n(8),b=n(139),j=n(140),u=["bsPrefix","active","disabled","className","variant","action","as","onClick"],h={variant:void 0,active:!1,disabled:!1},x=s.a.forwardRef((function(e,t){var n=e.bsPrefix,r=e.active,l=e.disabled,b=e.className,h=e.variant,x=e.action,O=e.as,m=e.onClick,f=Object(c.a)(e,u);n=Object(d.a)(n,"list-group-item");var p=Object(o.useCallback)((function(e){if(l)return e.preventDefault(),void e.stopPropagation();m&&m(e)}),[l,m]);return l&&void 0===f.tabIndex&&(f.tabIndex=-1,f["aria-disabled"]=!0),s.a.createElement(j.a,Object(a.a)({ref:t},f,{as:O||(x?f.href?"a":"button":"div"),onClick:p,className:i()(b,n,r&&"active",l&&"disabled",h&&n+"-"+h,x&&n+"-action")}))}));x.defaultProps=h,x.displayName="ListGroupItem";var O=x,m=["className","bsPrefix","variant","horizontal","as"],f={variant:void 0,horizontal:void 0},p=s.a.forwardRef((function(e,t){var n,r=Object(l.a)(e,{activeKey:"onSelect"}),o=r.className,j=r.bsPrefix,u=r.variant,h=r.horizontal,x=r.as,O=void 0===x?"div":x,f=Object(c.a)(r,m),p=Object(d.a)(j,"list-group");return n=h?!0===h?"horizontal":"horizontal-"+h:null,s.a.createElement(b.a,Object(a.a)({ref:t},f,{as:O,className:i()(o,p,u&&p+"-"+u,n&&p+"-"+n)}))}));p.defaultProps=f,p.displayName="ListGroup",p.Item=O;t.a=p},539:function(e,t,n){"use strict";n.r(t),n.d(t,"default",(function(){return p}));var a=n(20),c=n(533),r=n(191),i=n(188),o=n(127),s=n(73),l=n(189),d=n(532),b=n(49),j=n(120),u=n(19),h=n(42),x=n(40),O=n(14),m=n(1);function f(e){var t=e.text,n=e.path,a=void 0===n?"":n,r=Object(u.h)(),i=!!Object(u.f)(r.pathname,{path:"/".concat(a),exact:!0});return Object(m.jsx)(c.a.Item,{action:!0,variant:i?"customdark":"customlight",className:"text-white",as:h.b,to:"/".concat(a),children:i?Object(m.jsx)("strong",{children:t}):t})}function p(e){return b.a.pageview("/doc"),Object(m.jsx)(r.a,{className:"my-2",children:Object(m.jsx)(h.a,{basename:"/doc/",children:Object(m.jsxs)(i.a,{bg:"darkcontent",text:"lightfont",children:[Object(m.jsx)(i.a.Header,{children:Object(m.jsxs)(o.a,{children:[Object(m.jsx)(s.a,{children:Object(m.jsx)("h4",{className:"mb-0",children:"Documentation"})}),Object(m.jsx)(s.a,{xs:"auto",children:Object(m.jsx)("h4",{className:"mb-0",children:Object(m.jsx)(l.a,{variant:"info",children:"Version. 1"})})})]})}),Object(m.jsx)(i.a.Body,{children:Object(m.jsx)(d.a.Container,{children:Object(m.jsxs)(o.a,{children:[Object(m.jsx)(s.a,{md:3,children:Object(m.jsx)(i.a,{bg:"lightcontent",text:"lightfont",className:"h-100",children:Object(m.jsxs)(c.a,{className:"h-100",variant:"flush",children:[Object(m.jsx)(f,{text:"Overview"}),Object(m.jsx)(f,{text:Object(m.jsx)("code",{children:"StatKey"}),path:"StatKey"}),Object(m.jsx)(f,{text:Object(m.jsx)("code",{children:"ArtifactSetKey"}),path:"ArtifactSetKey"}),Object(m.jsx)(f,{text:Object(m.jsx)("code",{children:"CharacterKey"}),path:"CharacterKey"}),Object(m.jsx)(f,{text:Object(m.jsx)("code",{children:"WeaponKey"}),path:"WeaponKey"})]})})}),Object(m.jsx)(s.a,{md:9,children:Object(m.jsx)(i.a,{bg:"lightcontent",text:"lightfont",className:"h-100",children:Object(m.jsx)(i.a.Body,{children:Object(m.jsxs)(u.d,{children:[Object(m.jsx)(u.b,{path:"/ArtifactSetKey",component:y}),Object(m.jsx)(u.b,{path:"/WeaponKey",component:K}),Object(m.jsx)(u.b,{path:"/CharacterKey",component:E}),Object(m.jsx)(u.b,{path:"/StatKey",component:g}),Object(m.jsx)(u.b,{path:"/",component:v})]})})})})]})})})]})})})}function v(){return Object(m.jsxs)(m.Fragment,{children:[Object(m.jsx)("h4",{children:"Genshin Open Object Description (GOOD)"}),Object(m.jsxs)("div",{className:"mb-2",children:[Object(m.jsxs)("p",{children:[Object(m.jsx)("strong",{children:"GOOD"})," is a data format description to map Genshin Data into a parsable JSON. This is intended to be a standardized format to allow Genshin developers/programmers to transfer data without needing manual conversion."]}),Object(m.jsx)("p",{children:"As of version 6.0.0, Genshin Optimizer's database export conforms to this format."}),Object(m.jsx)(i.a,{bg:"darkcontent",text:"lightfont",children:Object(m.jsx)(i.a.Body,{children:Object(m.jsx)(C,{text:'interface IGOOD {\n  format: "GOOD" //A way for people to recognize this format.\n  version: number //API version.\n  source: string //the app that generates this data.\n  characters: ICharacter[]\n  artifacts: IArtifact[]\n  weapons: IWeapon[]\n}'})})})]}),Object(m.jsx)("br",{}),Object(m.jsx)("h4",{children:"Artifact data representation"}),Object(m.jsx)(i.a,{bg:"darkcontent",text:"lightfont",children:Object(m.jsx)(i.a.Body,{children:Object(m.jsx)(C,{text:'interface IArtifact {\n  setKey: SetKey //e.g. "GladiatorsFinale"\n  slotKey: SlotKey //e.g. "plume"\n  level: number //0-20 inclusive\n  rarity: number //3-5 inclusive\n  mainStatKey: StatKey\n  location: CharacterKey|"" //where "" means not equipped.\n  lock: boolean //Whether the artifact is locked in game.\n  substats: ISubstat[]\n}\n  \ninterface ISubstat {\n  key: StatKey //e.g. "critDMG_"\n  value: number //e.g. 19.4\n}'})})}),Object(m.jsx)("br",{}),Object(m.jsx)("h4",{children:"Weapon data representation"}),Object(m.jsx)(i.a,{bg:"darkcontent",text:"lightfont",children:Object(m.jsx)(i.a.Body,{children:Object(m.jsx)(C,{text:'interface IWeapon {\n  key: WeaponKey //"CrescentPike"\n  level: number //1-90 inclusive\n  ascension: number //0-6 inclusive. need to disambiguate 80/90 or 80/80\n  refinement: number //1-5 inclusive\n  location: CharacterKey | "" //where "" means not equipped.\n}'})})}),Object(m.jsx)("br",{}),Object(m.jsx)("h4",{children:"Character data representation"}),Object(m.jsx)(i.a,{bg:"darkcontent",text:"lightfont",children:Object(m.jsx)(i.a.Body,{children:Object(m.jsx)(C,{text:'interface ICharacter {\n  key: CharacterKey //e.g. "Rosaria"\n  level: number //1-90 inclusive\n  constellation: number //0-6 inclusive\n  ascension: number //0-6 inclusive. need to disambiguate 80/90 or 80/80\n  talent: { //does not include boost from constellations. 1-15 inclusive\n    auto: number\n    skill: number\n    burst: number\n  }\n}'})})})]})}function g(){var e="type StatKey\n  = ".concat(["hp","hp_","atk","atk_","def","def_","eleMas","enerRech_","heal_","critRate_","critDMG_","physical_dmg_","anemo_dmg_","geo_dmg_","electro_dmg_","hydro_dmg_","pyro_dmg_","cryo_dmg_"].map((function(e){var t;return'"'.concat(e,'" //').concat(null===(t=x.d[e])||void 0===t?void 0:t.name).concat((null===e||void 0===e?void 0:e.endsWith("_"))?"%":"")})).join("\n  | "));return Object(m.jsxs)(m.Fragment,{children:[Object(m.jsx)("h4",{children:"StatKey"}),Object(m.jsx)(i.a,{bg:"darkcontent",text:"lightfont",children:Object(m.jsx)(i.a.Body,{children:Object(m.jsx)(C,{text:e})})})]})}function y(){var e=Object(j.a)(Object(a.a)(new Set(O.b)).map((function(e){return"artifact_".concat(e,"_gen")}))).t,t="type ArtifactSetKey\n  = ".concat(Object(a.a)(new Set(O.b)).sort().map((function(t){return'"'.concat(t,'" //').concat(e("artifact_".concat(t,"_gen:setName")))})).join("\n  | "));return Object(m.jsxs)(m.Fragment,{children:[Object(m.jsx)("h4",{children:"ArtifactSetKey"}),Object(m.jsx)(i.a,{bg:"darkcontent",text:"lightfont",children:Object(m.jsx)(i.a.Body,{children:Object(m.jsx)(C,{text:t})})})]})}function E(){var e=Object(j.a)(Object(a.a)(new Set(O.c)).map((function(e){return"char_".concat(e,"_gen")}))).t,t="type CharacterKey\n  = ".concat(Object(a.a)(new Set(O.c)).sort().map((function(t){return'"'.concat(t,'" //').concat(e("char_".concat(t,"_gen:name")))})).join("\n  | "));return Object(m.jsxs)(m.Fragment,{children:[Object(m.jsx)("h4",{children:"CharacterKey"}),Object(m.jsx)(i.a,{bg:"darkcontent",text:"lightfont",children:Object(m.jsx)(i.a.Body,{children:Object(m.jsx)(C,{text:t})})})]})}function K(){var e=Object(j.a)(Object(a.a)(new Set(O.i)).map((function(e){return"weapon_".concat(e,"_gen")}))).t,t="type WeaponKey\n  = ".concat(Object(a.a)(new Set(O.i)).sort().map((function(t){return'"'.concat(t,'" //').concat(e("weapon_".concat(t,"_gen:name")))})).join("\n  | "));return Object(m.jsxs)(m.Fragment,{children:[Object(m.jsx)("h4",{children:"WeaponKey"}),Object(m.jsx)(i.a,{bg:"darkcontent",text:"lightfont",children:Object(m.jsx)(i.a.Body,{children:Object(m.jsx)(C,{text:t})})})]})}function C(e){var t=e.text,n=t.split(/\r\n|\r|\n/).length+1,a=Array.from(Array(n).keys()).map((function(e){return e+1})).join("\n");return Object(m.jsxs)("div",{className:"d-flex flex-row",children:[Object(m.jsx)("textarea",{className:"code text-secondary",disabled:!0,spellCheck:"false","aria-label":"Code Sample",rows:n,style:{width:"2em",overflow:"hidden",userSelect:"none"},value:a,unselectable:"off"}),Object(m.jsx)("textarea",{className:"code w-100 text-info flex-grow-1 ",disabled:!0,spellCheck:"false","aria-label":"Code Sample",rows:n,value:t})]})}}}]);
//# sourceMappingURL=20.bc604b84.chunk.js.map